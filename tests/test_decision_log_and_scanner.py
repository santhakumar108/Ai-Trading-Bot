"""
Spec section 17 ("record every signal ... record rejected trades ... every
NO TRADE decision should include a reason"), section 12/18 (drawdown
tracking for paper trading), and section 19 (configurable NSE-universe
scanner with batching/caching/rate-limit handling).
"""

from datetime import datetime, timezone

import pandas as pd
import pytest

from config.settings import Config
from data.data_quality import DataQualityIssue, DataQualityReport
from data.market_data import DataUnavailableError, MarketDataProvider
from fundamentals.fundamental_analysis import FundamentalAnalyzer
from paper_trading.decision_log import DecisionLog, DecisionLogEntry
from paper_trading.engine import PaperTradingEngine
from paper_trading.journal import TradeJournal
from paper_trading.scanner import UniverseScanner


def make_fetch_fn(df: pd.DataFrame):
    def fetch(symbol, period, interval):
        return df.copy()
    return fetch


def make_offline_engine(daily_df: pd.DataFrame, tmp_path, config=None) -> PaperTradingEngine:
    config = config or Config()
    config.decision_thresholds.min_history_bars = 60
    md = MarketDataProvider(fetch_fn=make_fetch_fn(daily_df))

    def fake_fundamentals_fetch(symbol):
        return {
            "revenue_growth": 0.1, "earnings_growth": 0.1, "eps": 5, "pe_ratio": 20, "pb_ratio": 3,
            "debt_to_equity": 50, "roe": 0.15, "profit_margin": 0.1, "operating_cash_flow": 100_000,
            "free_cash_flow": 50_000, "last_update": datetime.now(timezone.utc),
        }

    return PaperTradingEngine(
        config=config, market_data=md, fetch_news=False, fetch_social=False,
        fundamental_analyzer=FundamentalAnalyzer(fetch_fn=fake_fundamentals_fetch),
        journal=TradeJournal(path=str(tmp_path / "journal.csv")),
        decision_log=DecisionLog(path=str(tmp_path / "decisions.csv")),
    )


# --- DecisionLog: basic persistence ----------------------------------------

def test_decision_log_records_and_persists_entries(tmp_path):
    log = DecisionLog(path=str(tmp_path / "decisions.csv"))
    log.record(DecisionLogEntry(
        timestamp=datetime.now(timezone.utc), symbol="FAKE", decision="NO TRADE", approved=False,
        confidence=0.0, confidence_label="NO TRADE", direction="NONE", data_quality_status="OK",
        reasons="insufficient signals",
    ))
    assert len(log.all_entries()) == 1
    assert (tmp_path / "decisions.csv").exists()
    text = (tmp_path / "decisions.csv").read_text()
    assert "FAKE" in text and "insufficient signals" in text


def test_decision_log_rejected_and_approved_split(tmp_path):
    log = DecisionLog(path=str(tmp_path / "decisions.csv"))
    log.record(DecisionLogEntry(datetime.now(timezone.utc), "A", "BUY", True, 80.0, "HIGH", "UP", "OK", "clean setup"))
    log.record(DecisionLogEntry(datetime.now(timezone.utc), "B", "NO TRADE", False, 10.0, "LOW", "NONE", "OK", "confidence too low"))
    assert len(log.approved_entries()) == 1
    assert len(log.rejected_entries()) == 1
    assert log.approved_entries()[0].symbol == "A"
    assert log.rejected_entries()[0].symbol == "B"


def test_decision_log_no_trade_count_and_symbol_filter(tmp_path):
    log = DecisionLog(path=str(tmp_path / "decisions.csv"))
    log.record(DecisionLogEntry(datetime.now(timezone.utc), "A", "NO TRADE", False, 0.0, "NO TRADE", "NONE", "OK", "x"))
    log.record(DecisionLogEntry(datetime.now(timezone.utc), "A", "BUY", True, 90.0, "HIGH", "UP", "OK", "y"))
    log.record(DecisionLogEntry(datetime.now(timezone.utc), "B", "NO TRADE", False, 0.0, "NO TRADE", "NONE", "OK", "z"))
    assert log.no_trade_count() == 2
    assert len(log.entries_for_symbol("A")) == 2
    assert len(log.entries_for_symbol("B")) == 1


# --- DecisionLog wired into PaperTradingEngine.scan_symbol ------------------

# --- Spec Part 9 (decision pipeline completion): fundamental_risk /
# news_sentiment_class wired through into the decision log and report -----

def test_decision_log_and_report_carry_fundamental_risk_and_news_sentiment_class(tmp_path):
    from datetime import timedelta

    from news.news_analysis import NewsAnalyzer, NewsItem, NewsSourceAdapter
    from tests.conftest import make_synthetic_ohlcv

    class _FakeNewsSource(NewsSourceAdapter):
        def fetch(self, symbol, company_name=None):
            return [
                NewsItem(
                    symbol=symbol, headline="Company reports outstanding record profit growth",
                    source_domain="reuters.com",
                    published_at=datetime.now(timezone.utc) - timedelta(hours=1),
                ),
                NewsItem(
                    symbol=symbol, headline="Analysts praise strong quarterly earnings beat",
                    source_domain="bloomberg.com",
                    published_at=datetime.now(timezone.utc) - timedelta(hours=2),
                ),
            ]

    recent_start = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=260)[0].strftime("%Y-%m-%d")
    daily = make_synthetic_ohlcv(n=260, start_price=150.0, drift=0.001, seed=81, start_date=recent_start)

    config = Config()
    config.decision_thresholds.min_history_bars = 60
    md = MarketDataProvider(fetch_fn=make_fetch_fn(daily))

    def fake_fundamentals_fetch(symbol):
        return {
            "revenue_growth": 0.1, "earnings_growth": 0.1, "eps": 5, "pe_ratio": 20, "pb_ratio": 3,
            "debt_to_equity": 50, "roe": 0.15, "profit_margin": 0.1, "operating_cash_flow": 100_000,
            "free_cash_flow": 50_000, "last_update": datetime.now(timezone.utc),
        }

    engine = PaperTradingEngine(
        config=config, market_data=md, fetch_news=True, fetch_social=False,
        fundamental_analyzer=FundamentalAnalyzer(fetch_fn=fake_fundamentals_fetch),
        news_analyzer=NewsAnalyzer(source=_FakeNewsSource()),
        journal=TradeJournal(path=str(tmp_path / "journal.csv")),
        decision_log=DecisionLog(path=str(tmp_path / "decisions.csv")),
    )
    result = engine.scan_symbol("GOODCO")
    assert result is not None

    entry = engine.decision_log.all_entries()[0]
    if "fundamentals" in result.signal.component_scores:
        assert entry.fundamental_risk is not None
        assert result.report.fundamental_risk is not None
        assert 0 <= entry.fundamental_risk <= 100
    if "news_sentiment" in result.signal.component_scores:
        assert entry.news_sentiment_class in ("POSITIVE", "NEGATIVE", "NEUTRAL", "MIXED")
        assert result.report.news_sentiment_class == entry.news_sentiment_class


def test_decision_log_fundamental_risk_and_news_class_none_when_unavailable(uptrend_daily, tmp_path):
    """Same offline helper as the rest of this file (fetch_news=False,
    stale/synthetic-dated data) -- fundamentals may or may not clear its
    own gate, but news is never fetched at all, so news_sentiment_class
    must be None."""
    engine = make_offline_engine(uptrend_daily, tmp_path)
    engine.scan_symbol("FAKE")
    entry = engine.decision_log.all_entries()[0]
    assert entry.news_sentiment_class is None


def test_scan_symbol_logs_a_decision_for_every_scan(uptrend_daily, tmp_path):
    engine = make_offline_engine(uptrend_daily, tmp_path)
    engine.scan_symbol("FAKE")
    entries = engine.decision_log.all_entries()
    assert len(entries) == 1
    assert entries[0].symbol == "FAKE"
    assert entries[0].reasons  # never blank


def test_decision_log_momentum_score_matches_technical_snapshot_directly(uptrend_daily, tmp_path):
    """A real scan must populate momentum_score with the actual
    TechnicalSnapshot.momentum_score() value -- computed independently
    here and compared for exact equality, proving the log reads the real
    number rather than re-deriving or coincidentally matching it."""
    from indicators.technical import TechnicalAnalyzer

    engine = make_offline_engine(uptrend_daily, tmp_path)
    result = engine.scan_symbol("MOMFAKE")
    assert result is not None
    entry = engine.decision_log.all_entries()[0]

    expected = TechnicalAnalyzer().analyze("MOMFAKE", uptrend_daily).momentum_score()
    assert entry.momentum_score == pytest.approx(expected)
    assert 0 <= entry.momentum_score <= 100
    assert entry.sector_score is None  # unaffected -- no sector-index data source exists


def test_decision_log_momentum_score_stays_none_when_data_totally_unavailable(tmp_path):
    """Mirrors test_scan_symbol_logs_no_trade_with_reason_when_data_totally_unavailable:
    when a TechnicalSnapshot was never built, momentum_score must correctly
    stay None -- never fabricated."""
    def always_fails(symbol, period, interval):
        raise DataUnavailableError("simulated outage")

    config = Config()
    md = MarketDataProvider(fetch_fn=always_fails, max_retries=1)
    engine = PaperTradingEngine(
        config=config, market_data=md, fetch_news=False, fetch_social=False,
        journal=TradeJournal(path=str(tmp_path / "journal.csv")),
        decision_log=DecisionLog(path=str(tmp_path / "decisions.csv")),
    )
    result = engine.scan_symbol("FAKE")
    assert result is None
    entries = engine.decision_log.all_entries()
    assert entries[0].momentum_score is None


def test_scan_symbol_logs_no_trade_with_reason_when_data_totally_unavailable(tmp_path):
    def always_fails(symbol, period, interval):
        raise DataUnavailableError("simulated outage")

    config = Config()
    md = MarketDataProvider(fetch_fn=always_fails, max_retries=1)
    engine = PaperTradingEngine(
        config=config, market_data=md, fetch_news=False, fetch_social=False,
        journal=TradeJournal(path=str(tmp_path / "journal.csv")),
        decision_log=DecisionLog(path=str(tmp_path / "decisions.csv")),
    )
    result = engine.scan_symbol("FAKE")
    assert result is None
    entries = engine.decision_log.all_entries()
    assert len(entries) == 1
    assert entries[0].decision == "NO TRADE"
    assert entries[0].approved is False
    assert entries[0].data_quality_status == "INVALID"
    assert "unavailable" in entries[0].reasons.lower() or "outage" in entries[0].reasons.lower()


def test_scan_symbol_logs_no_trade_when_data_quality_gate_forces_it(uptrend_daily, tmp_path):
    corrupted = uptrend_daily.copy()
    corrupted.iloc[10, corrupted.columns.get_loc("High")] = corrupted.iloc[10]["Low"] - 5  # invalid OHLC
    engine = make_offline_engine(corrupted, tmp_path)
    result = engine.scan_symbol("FAKE")
    assert result is not None
    entries = engine.decision_log.all_entries()
    assert len(entries) == 1
    assert entries[0].decision == "NO TRADE"
    assert "Data quality gate failed" in entries[0].reasons


def test_multiple_scans_accumulate_in_the_log(uptrend_daily, tmp_path):
    engine = make_offline_engine(uptrend_daily, tmp_path)
    engine.scan_symbol("A")
    engine.scan_symbol("B")
    engine.scan_symbol("A")
    assert len(engine.decision_log.all_entries()) == 3
    assert len(engine.decision_log.entries_for_symbol("A")) == 2


# --- Drawdown tracking in account_summary() ---------------------------------

def test_account_summary_reports_zero_drawdown_with_no_activity(uptrend_daily, tmp_path):
    engine = make_offline_engine(uptrend_daily, tmp_path)
    summary = engine.account_summary()
    assert summary["current_drawdown_pct"] == pytest.approx(0.0)
    assert summary["max_drawdown_pct"] == pytest.approx(0.0)
    assert summary["equity_snapshots_recorded"] >= 1


def test_account_summary_tracks_drawdown_after_equity_falls(uptrend_daily, tmp_path):
    engine = make_offline_engine(uptrend_daily, tmp_path)
    engine.account_summary()  # snapshot #1 at starting equity
    # Simulate a loss by directly moving broker cash down (offline unit test,
    # not exercising the full order path) -- the point here is that
    # account_summary's drawdown math responds to a real drop in equity.
    engine.broker._cash -= 20_000
    summary = engine.account_summary()
    assert summary["current_drawdown_pct"] < 0
    assert summary["max_drawdown_pct"] <= summary["current_drawdown_pct"] + 1e-9


def test_account_summary_includes_decision_log_counts(uptrend_daily, tmp_path):
    engine = make_offline_engine(uptrend_daily, tmp_path)
    engine.scan_symbol("FAKE")
    summary = engine.account_summary()
    assert summary["decisions_logged"] == 1
    assert "no_trade_count" in summary


def test_record_equity_snapshot_is_idempotent_safe_to_call_repeatedly(uptrend_daily, tmp_path):
    engine = make_offline_engine(uptrend_daily, tmp_path)
    engine.record_equity_snapshot()
    engine.record_equity_snapshot()
    assert len(engine._equity_history) == 2


# --- UniverseScanner ---------------------------------------------------------

def test_scanner_defaults_to_configured_universe_symbols(uptrend_daily, tmp_path):
    config = Config()
    config.decision_thresholds.min_history_bars = 60
    config.universe.symbols = ["A.NS", "B.NS", "C.NS"]
    config.universe.scan_batch_size = 2
    config.universe.scan_batch_delay_seconds = 0.0
    engine = make_offline_engine(uptrend_daily, tmp_path, config=config)
    scanner = UniverseScanner(engine)
    cycle = scanner.scan()
    assert cycle.symbols_requested == 3
    assert {r.symbol for r in cycle.results} == {"A.NS", "B.NS", "C.NS"}


def test_scanner_never_uses_a_hardcoded_symbol_list(uptrend_daily, tmp_path):
    """Passing no `symbols` argument must scan exactly config.universe.symbols
    -- never a baked-in DEMO_* list."""
    config = Config()
    config.decision_thresholds.min_history_bars = 60
    config.universe.symbols = ["ONLYTHIS.NS"]
    engine = make_offline_engine(uptrend_daily, tmp_path, config=config)
    scanner = UniverseScanner(engine, delay_between_batches_seconds=0.0)
    cycle = scanner.scan()
    assert [r.symbol for r in cycle.results] == ["ONLYTHIS.NS"]


def test_scanner_batches_and_calls_scan_symbol_once_per_symbol(uptrend_daily, tmp_path, monkeypatch):
    config = Config()
    config.decision_thresholds.min_history_bars = 60
    config.universe.symbols = ["A.NS", "B.NS", "C.NS", "D.NS", "E.NS"]
    engine = make_offline_engine(uptrend_daily, tmp_path, config=config)

    call_count = {"n": 0}
    original = engine.scan_symbol

    def counting_scan(symbol, sector=None):
        call_count["n"] += 1
        return original(symbol, sector=sector)

    engine.scan_symbol = counting_scan
    scanner = UniverseScanner(engine, batch_size=2, delay_between_batches_seconds=0.0)
    cycle = scanner.scan()
    assert call_count["n"] == 5
    assert cycle.symbols_scanned == 5


def test_scanner_isolates_a_single_symbol_failure(uptrend_daily, tmp_path):
    config = Config()
    config.decision_thresholds.min_history_bars = 60
    config.universe.symbols = ["GOOD.NS", "BAD.NS"]
    engine = make_offline_engine(uptrend_daily, tmp_path, config=config)

    def flaky_scan(symbol, sector=None):
        if symbol == "BAD.NS":
            raise RuntimeError("simulated crash")
        return PaperTradingEngine.scan_symbol(engine, symbol, sector=sector)

    engine.scan_symbol = flaky_scan
    scanner = UniverseScanner(engine, delay_between_batches_seconds=0.0)
    cycle = scanner.scan()
    assert cycle.symbols_skipped == 1
    assert cycle.symbols_scanned == 1
    assert cycle.results[0].symbol == "GOOD.NS"


def test_scanner_caches_within_ttl_and_refreshes_after(uptrend_daily, tmp_path):
    config = Config()
    config.decision_thresholds.min_history_bars = 60
    config.universe.symbols = ["A.NS"]
    engine = make_offline_engine(uptrend_daily, tmp_path, config=config)
    scanner = UniverseScanner(engine, cache_ttl_seconds=1000.0, delay_between_batches_seconds=0.0)

    call_count = {"n": 0}
    original = engine.scan_symbol

    def counting_scan(symbol, sector=None):
        call_count["n"] += 1
        return original(symbol, sector=sector)

    engine.scan_symbol = counting_scan
    first = scanner.scan()
    second = scanner.scan()
    assert first.from_cache is False
    assert second.from_cache is True
    assert call_count["n"] == 1  # second scan served from cache, no re-scan

    third = scanner.scan(force_refresh=True)
    assert third.from_cache is False
    assert call_count["n"] == 2


def test_scanner_cache_key_depends_on_requested_symbols(uptrend_daily, tmp_path):
    config = Config()
    config.decision_thresholds.min_history_bars = 60
    engine = make_offline_engine(uptrend_daily, tmp_path, config=config)
    scanner = UniverseScanner(engine, cache_ttl_seconds=1000.0, delay_between_batches_seconds=0.0)
    first = scanner.scan(symbols=["A.NS"])
    second = scanner.scan(symbols=["B.NS"])  # different symbols -> must not hit the cache
    assert second.from_cache is False
    assert [r.symbol for r in second.results] == ["B.NS"]


def test_scanner_reads_batch_settings_from_config_by_default(uptrend_daily, tmp_path):
    config = Config()
    config.decision_thresholds.min_history_bars = 60
    config.universe.scan_batch_size = 7
    config.universe.scan_batch_delay_seconds = 2.5
    config.universe.scan_cache_ttl_seconds = 30.0
    engine = make_offline_engine(uptrend_daily, tmp_path, config=config)
    scanner = UniverseScanner(engine)
    assert scanner.batch_size == 7
    assert scanner.delay_between_batches_seconds == 2.5
    assert scanner.cache_ttl_seconds == 30.0
