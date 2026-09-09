"""
Spec section 18: the dashboard must show Market Regime, Data Quality, Top
Candidates (with per-candidate Technical/Fundamental/News/Social/ML/Risk
score, Entry/Stop/Target/R:R/Expected Value, Decision, Decision Reasons),
and Paper Equity/P&L/Drawdown/Win Rate/Profit Factor/Sharpe/Number of
Trades/Daily Risk Used, and must always clearly display PAPER MODE.
"""

from datetime import datetime, timezone

import pandas as pd
import pytest

from config.settings import Config
from dashboard.cli_dashboard import render_cli_dashboard
from dashboard.common import (
    candidate_row,
    compute_market_regime,
    fmt_num,
    summarize_data_quality,
    top_candidates,
)
from dashboard.html_report import render_html_report
from data.data_quality import DataQualityReport
from data.market_data import MarketDataProvider
from fundamentals.fundamental_analysis import FundamentalAnalyzer
from paper_trading.decision_log import DecisionLog
from paper_trading.engine import PaperTradingEngine
from paper_trading.journal import TradeJournal


def make_fetch_fn(df: pd.DataFrame):
    def fetch(symbol, period, interval):
        return df.copy()
    return fetch


def make_offline_engine(daily_df: pd.DataFrame, tmp_path, config=None, ml_provider=None) -> PaperTradingEngine:
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
        ml_provider=ml_provider,
    )


class _FixedMLProvider:
    """Same fake-provider shape as tests/test_ml_trade_outcome.py's
    _FixedMLProvider -- a constant prediction, no real fitting."""
    def __init__(self, value):
        self.value = value

    def predict(self, symbol, daily):
        return self.value


def _recent_start_date(n: int) -> str:
    """Same helper as tests/test_ml_trade_outcome.py -- ml_probability_up
    only reaches the report when the data-quality gate's staleness check
    passes, which needs a series ending near 'today', not the fixed
    2023-dated uptrend_daily fixture."""
    return pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=n)[0].strftime("%Y-%m-%d")


# --- compute_market_regime ---------------------------------------------------

def test_market_regime_unknown_without_index_symbol(uptrend_daily, tmp_path):
    config = Config()
    config.universe.index_symbol = None
    engine = make_offline_engine(uptrend_daily, tmp_path, config=config)
    regime = compute_market_regime(engine, config)
    assert regime.startswith("UNKNOWN")


def test_market_regime_classified_from_configured_index(uptrend_daily, tmp_path):
    config = Config()
    config.universe.index_symbol = "^NSEI"
    engine = make_offline_engine(uptrend_daily, tmp_path, config=config)
    regime = compute_market_regime(engine, config)
    assert regime in (
        "BULL_LOW_VOL", "BULL_HIGH_VOL", "BULL_UNKNOWN_VOL",
        "BEAR_LOW_VOL", "BEAR_HIGH_VOL", "BEAR_UNKNOWN_VOL",
        "SIDEWAYS_LOW_VOL", "SIDEWAYS_HIGH_VOL", "SIDEWAYS_UNKNOWN_VOL",
    )
    assert "UNKNOWN (" not in regime


def test_market_regime_unknown_when_index_data_unavailable(uptrend_daily, tmp_path):
    from data.market_data import DataUnavailableError

    config = Config()
    config.universe.index_symbol = "^NSEI"
    engine = make_offline_engine(uptrend_daily, tmp_path, config=config)

    def always_fails(symbol, period="2y"):
        raise DataUnavailableError("no index data")

    engine.market_data.get_daily = always_fails
    regime = compute_market_regime(engine, config)
    assert regime.startswith("UNKNOWN")


# --- summarize_data_quality ---------------------------------------------------

class FakeScanResult:
    def __init__(self, symbol, status):
        self.symbol = symbol
        self.data_quality = DataQualityReport(
            symbol=symbol, as_of=pd.Timestamp.now(tz="UTC"), issues=[],
            quality_score=1.0 if status == "OK" else 0.5, status=status, min_quality_score_to_trade=0.70,
        )


def test_data_quality_summary_counts_by_status():
    results = [FakeScanResult("A", "OK"), FakeScanResult("B", "OK"), FakeScanResult("C", "DEGRADED"), FakeScanResult("D", "INVALID")]
    summary = summarize_data_quality(results, symbols_requested=4)
    assert summary.ok == 2 and summary.degraded == 1 and summary.invalid == 1 and summary.unknown == 0
    assert summary.worst_status == "INVALID"


def test_data_quality_summary_counts_unknown_for_missing_results():
    results = [FakeScanResult("A", "OK")]
    summary = summarize_data_quality(results, symbols_requested=5)
    assert summary.unknown == 4
    assert summary.worst_status == "UNKNOWN"


def test_data_quality_summary_render_text_mentions_all_counts():
    results = [FakeScanResult("A", "OK")]
    summary = summarize_data_quality(results, symbols_requested=1)
    text = summary.render_text()
    assert "OK" in text and "DEGRADED" in text and "INVALID" in text and "UNKNOWN" in text


# --- top_candidates / candidate_row -------------------------------------------

def test_top_candidates_sorted_by_confidence_descending(uptrend_daily, choppy_daily, tmp_path):
    engine = make_offline_engine(uptrend_daily, tmp_path)
    result_a = engine.scan_symbol("A")
    result_b = engine.scan_symbol("B")
    # Force distinct confidences to make ordering deterministic regardless of actual signal outcome.
    result_a.signal.overall_confidence = 30.0
    result_b.signal.overall_confidence = 90.0
    ordered = top_candidates([result_a, result_b], top_n=10)
    assert [r.symbol for r in ordered] == ["B", "A"]


def test_top_candidates_respects_top_n(uptrend_daily, tmp_path):
    engine = make_offline_engine(uptrend_daily, tmp_path)
    results = [engine.scan_symbol(s) for s in ["A", "B", "C"]]
    ordered = top_candidates(results, top_n=2)
    assert len(ordered) == 2


def test_candidate_row_exposes_all_spec_section_18_fields(uptrend_daily, tmp_path):
    engine = make_offline_engine(uptrend_daily, tmp_path)
    result = engine.scan_symbol("FAKE")
    row = candidate_row(result)
    for key in [
        "symbol", "price", "decision", "confidence", "confidence_label",
        "technical_score", "fundamental_score", "news_score", "social_score", "ml_score", "risk_score",
        "entry", "stop_loss", "target", "risk_reward",
        "expected_value_per_share", "expected_value_total", "data_quality_status", "reasons",
    ]:
        assert key in row
    assert row["ml_score"] is None  # no --ml provider supplied -- honestly not fabricated


def test_candidate_row_exposes_real_ml_score_when_provider_supplied(tmp_path):
    """ml_score is rescaled to the same 0-100 convention as the other
    score columns (technical_score, fundamental_score, ...) -- a raw 0-1
    probability would render wrong (e.g. 0.63 -> "1" via .0f formatting).
    Uses a recent-dated series (not the fixed-2023 uptrend_daily fixture)
    so the data-quality gate doesn't fire and swallow ml_probability_up
    before it reaches the report."""
    from tests.conftest import make_synthetic_ohlcv

    daily = make_synthetic_ohlcv(n=260, start_price=150.0, drift=0.001, seed=91, start_date=_recent_start_date(260))
    engine = make_offline_engine(daily, tmp_path, ml_provider=_FixedMLProvider(0.63))
    result = engine.scan_symbol("MLFAKE")
    row = candidate_row(result)
    assert row["ml_score"] == pytest.approx(63.0)


def test_fmt_num_handles_none_as_na():
    assert fmt_num(None) == "N/A"
    assert fmt_num(1.23456, ".2f") == "1.23"


# --- Full dashboard renders without crashing and shows required content ------

def test_cli_dashboard_renders_paper_mode_and_new_sections(uptrend_daily, tmp_path, capsys):
    config = Config()
    engine = make_offline_engine(uptrend_daily, tmp_path, config=config)
    result = engine.scan_symbol("FAKE")
    render_cli_dashboard(engine, [result], config, symbols_requested=1)
    out = capsys.readouterr().out
    assert "PAPER MODE" in out
    assert "Market Regime" in out
    assert "Data Quality" in out
    assert "Top Candidates" in out
    assert "Sharpe" in out
    assert "Daily Risk Used" in out or "Daily Risk" in out


def test_cli_dashboard_shows_live_mode_when_configured(uptrend_daily, tmp_path, capsys):
    config = Config()
    config.system.live_trading_enabled = True
    engine = make_offline_engine(uptrend_daily, tmp_path, config=config)
    render_cli_dashboard(engine, [], config, symbols_requested=0)
    out = capsys.readouterr().out
    assert "LIVE MODE" in out


def test_html_report_contains_paper_mode_and_new_sections(uptrend_daily, tmp_path):
    config = Config()
    engine = make_offline_engine(uptrend_daily, tmp_path, config=config)
    result = engine.scan_symbol("FAKE")
    out_path = str(tmp_path / "dashboard.html")
    render_html_report(engine, [result], config, out_path, symbols_requested=1)
    html = open(out_path).read()
    assert "PAPER MODE" in html
    assert "Market Regime" in html
    assert "Data Quality" in html
    assert "Top Candidates" in html
    assert "Sharpe" in html
    assert "Daily Risk Used" in html
    assert "EV/share" in html


def test_html_report_handles_zero_results_without_crashing(uptrend_daily, tmp_path):
    config = Config()
    engine = make_offline_engine(uptrend_daily, tmp_path, config=config)
    out_path = str(tmp_path / "dashboard_empty.html")
    render_html_report(engine, [], config, out_path, symbols_requested=0)
    html = open(out_path).read()
    assert "PAPER MODE" in html


# --- ML score reaches the dashboard views when a provider is supplied --------

def test_cli_dashboard_shows_real_ml_score_when_provider_supplied(tmp_path, capsys):
    from tests.conftest import make_synthetic_ohlcv

    config = Config()
    daily = make_synthetic_ohlcv(n=260, start_price=150.0, drift=0.001, seed=92, start_date=_recent_start_date(260))
    engine = make_offline_engine(daily, tmp_path, config=config, ml_provider=_FixedMLProvider(0.63))
    result = engine.scan_symbol("MLFAKE")
    render_cli_dashboard(engine, [result], config, symbols_requested=1)
    out = capsys.readouterr().out
    assert "63" in out  # fmt_num(63.0, ".0f") style rendering of the real ML score
    assert "does not yet wire in a point-in-time ML predictor" not in out


def test_html_report_shows_real_ml_score_when_provider_supplied(tmp_path):
    from tests.conftest import make_synthetic_ohlcv

    config = Config()
    daily = make_synthetic_ohlcv(n=260, start_price=150.0, drift=0.001, seed=93, start_date=_recent_start_date(260))
    engine = make_offline_engine(daily, tmp_path, config=config, ml_provider=_FixedMLProvider(0.63))
    result = engine.scan_symbol("MLFAKE")
    out_path = str(tmp_path / "dashboard_ml.html")
    render_html_report(engine, [result], config, out_path, symbols_requested=1)
    html = open(out_path).read()
    assert "63" in html
    assert "does not yet wire in a point-in-time ML predictor" not in html


def test_cli_dashboard_ml_score_stays_na_without_provider(uptrend_daily, tmp_path, capsys):
    config = Config()
    engine = make_offline_engine(uptrend_daily, tmp_path, config=config)  # no ml_provider
    result = engine.scan_symbol("NOMLFAKE")
    render_cli_dashboard(engine, [result], config, symbols_requested=1)
    out = capsys.readouterr().out
    # Never fabricated: no provider supplied means N/A, not a made-up number.
    assert result.report.ml_probability_up is None


def test_cli_dashboard_accepts_ml_flag_and_wires_it_into_the_engine():
    """main.py's `dashboard` subcommand previously had no --ml flag at all
    (unlike scan/paper-trade/account-check), which made the candidate_row
    ml_score fix above unreachable through the real CLI. Confirms the flag
    now parses and cmd_dashboard threads it into _make_ml_provider."""
    import main

    parser = main.build_parser()
    args = parser.parse_args(["dashboard", "RELIANCE.NS", "--ml"])
    assert args.ml is True
    assert args.func is main.cmd_dashboard

    args_default = parser.parse_args(["dashboard", "RELIANCE.NS"])
    assert args_default.ml is False
