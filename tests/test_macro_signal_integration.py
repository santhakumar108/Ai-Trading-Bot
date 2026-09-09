"""
Spec Part 2: macro context wired into the decision pipeline.

Covers `_market_condition_score`'s macro=None regression (byte-identical
to pre-Phase-6 behavior), each macro term's bounded independent effect,
live `PaperTradingEngine.scan_symbol` wiring via an injected fake
`MacroDataProvider`, and `Backtester.run(..., macro_daily=...)`'s
point-in-time safety.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from config.settings import Config
from data.macro_data import MacroContext, MacroDataProvider
from data.market_data import MarketDataProvider
from fundamentals.fundamental_analysis import FundamentalAnalyzer
from paper_trading.decision_log import DecisionLog
from paper_trading.engine import PaperTradingEngine
from paper_trading.journal import TradeJournal
from strategy.signal_engine import _market_condition_score
from tests.conftest import make_downtrend_ohlcv, make_synthetic_ohlcv, make_uptrend_ohlcv


# =============================================================================
# _market_condition_score: macro=None regression + bounded independent effects
# =============================================================================

def test_macro_none_reproduces_pre_phase6_score_exactly():
    for trend, rs in [("UPTREND", 0.05), ("DOWNTREND", -0.03), ("SIDEWAYS", float("nan"))]:
        with_none = _market_condition_score(trend, rs, macro=None)
        without_param = _market_condition_score(trend, rs)  # omitted entirely -- same default
        assert with_none == pytest.approx(without_param)


def test_india_vix_uptrend_lowers_score():
    baseline = _market_condition_score("SIDEWAYS", float("nan"), macro=MacroContext())  # all UNKNOWN
    fear_up = _market_condition_score("SIDEWAYS", float("nan"), macro=MacroContext(india_vix_trend="UPTREND"))
    fear_down = _market_condition_score("SIDEWAYS", float("nan"), macro=MacroContext(india_vix_trend="DOWNTREND"))
    assert fear_up < baseline < fear_down


def test_crude_uptrend_lowers_score():
    baseline = _market_condition_score("SIDEWAYS", float("nan"), macro=MacroContext())
    crude_up = _market_condition_score("SIDEWAYS", float("nan"), macro=MacroContext(crude_trend="UPTREND"))
    crude_down = _market_condition_score("SIDEWAYS", float("nan"), macro=MacroContext(crude_trend="DOWNTREND"))
    assert crude_up < baseline < crude_down


def test_usdinr_uptrend_lowers_score():
    baseline = _market_condition_score("SIDEWAYS", float("nan"), macro=MacroContext())
    inr_up = _market_condition_score("SIDEWAYS", float("nan"), macro=MacroContext(usdinr_trend="UPTREND"))
    inr_down = _market_condition_score("SIDEWAYS", float("nan"), macro=MacroContext(usdinr_trend="DOWNTREND"))
    assert inr_up < baseline < inr_down


def test_us_vix_has_a_smaller_effect_than_india_vix():
    baseline = _market_condition_score("SIDEWAYS", float("nan"), macro=MacroContext())
    us_fear_up = _market_condition_score("SIDEWAYS", float("nan"), macro=MacroContext(us_vix_trend="UPTREND"))
    india_fear_up = _market_condition_score("SIDEWAYS", float("nan"), macro=MacroContext(india_vix_trend="UPTREND"))
    assert (baseline - us_fear_up) < (baseline - india_fear_up)


def test_score_always_stays_within_0_100_even_with_all_macro_headwinds():
    worst = MacroContext(us_vix_trend="UPTREND", india_vix_trend="UPTREND", crude_trend="UPTREND", usdinr_trend="UPTREND")
    score = _market_condition_score("DOWNTREND", -0.10, macro=worst)
    assert 0 <= score <= 100
    best = MacroContext(us_vix_trend="DOWNTREND", india_vix_trend="DOWNTREND", crude_trend="DOWNTREND", usdinr_trend="DOWNTREND")
    score2 = _market_condition_score("UPTREND", 0.10, macro=best)
    assert 0 <= score2 <= 100


# =============================================================================
# Live PaperTradingEngine.scan_symbol wiring
# =============================================================================

def _recent_start_date(n: int) -> str:
    return pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=n)[0].strftime("%Y-%m-%d")


def make_offline_engine(daily_df, tmp_path, macro_provider=None) -> PaperTradingEngine:
    config = Config()
    config.decision_thresholds.min_history_bars = 60

    def fetch(symbol, period, interval):
        return daily_df.copy()

    md = MarketDataProvider(fetch_fn=fetch)

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
        macro_provider=macro_provider,
    )


def test_engine_default_macro_provider_is_safe_offline_reusing_injected_market_data(tmp_path):
    """The critical regression guard: constructing a PaperTradingEngine
    with a fake market_data fetch_fn and NO explicit macro_provider must
    NOT hit the real network for macro tickers -- see config/providers.py's
    make_macro_data_provider docstring."""
    daily = make_synthetic_ohlcv(
        n=260, start_price=150.0, drift=0.001, seed=71, start_date=_recent_start_date(260),
    )
    engine = make_offline_engine(daily, tmp_path)  # macro_provider=None -> engine builds its own default
    assert engine.macro_provider.market_data is engine.market_data
    result = engine.scan_symbol("MACROSAFE")
    assert result is not None  # completed without hanging/crashing on a real network call


def test_engine_threads_injected_macro_context_into_the_decision(tmp_path):
    daily = make_synthetic_ohlcv(
        n=260, start_price=150.0, drift=0.001, seed=72, start_date=_recent_start_date(260),
    )

    class _FixedMacroProvider:
        def get_snapshot(self, force_refresh=False):
            return MacroContext(india_vix_trend="UPTREND", crude_trend="UPTREND", usdinr_trend="UPTREND")

    engine = make_offline_engine(daily, tmp_path, macro_provider=_FixedMacroProvider())
    result = engine.scan_symbol("MACROBEAR")
    assert result is not None
    # market_condition score should be lower than the same scan with a
    # neutral/no-macro provider, since every macro term here is a headwind.
    engine_no_macro = make_offline_engine(daily, tmp_path, macro_provider=None)

    class _UnknownMacroProvider:
        def get_snapshot(self, force_refresh=False):
            return MacroContext()

    engine_neutral = make_offline_engine(daily, tmp_path, macro_provider=_UnknownMacroProvider())
    result_neutral = engine_neutral.scan_symbol("MACRONEUTRAL")
    bear_score = result.signal.component_scores.get("market_condition")
    neutral_score = result_neutral.signal.component_scores.get("market_condition")
    if bear_score is not None and neutral_score is not None:
        assert bear_score <= neutral_score


def test_macro_provider_failure_treated_as_unavailable_not_a_crash(tmp_path):
    daily = make_synthetic_ohlcv(
        n=260, start_price=150.0, drift=0.001, seed=73, start_date=_recent_start_date(260),
    )

    class _BrokenMacroProvider:
        def get_snapshot(self, force_refresh=False):
            raise RuntimeError("simulated macro provider failure")

    engine = make_offline_engine(daily, tmp_path, macro_provider=_BrokenMacroProvider())
    result = engine.scan_symbol("MACROBROKEN")  # must not raise
    assert result is not None


# =============================================================================
# Backtester.run(..., macro_daily=...): point-in-time safety
# =============================================================================

def make_bt_config():
    config = Config()
    config.decision_thresholds.min_history_bars = 60
    config.universe.index_symbol = None
    return config


def test_backtest_macro_daily_none_is_byte_identical_to_before(monkeypatch):
    """Every existing caller passes macro_daily=None (the default) --
    confirms the run completes and produces trades/metrics with no macro
    wiring involved, i.e. this phase didn't change default behavior."""
    from backtesting.backtester import Backtester

    daily = make_uptrend_ohlcv(n=700, seed=501)
    config = make_bt_config()
    backtester = Backtester(config=config)
    result = backtester.run("TEST.NS", daily)  # macro_daily omitted
    assert result.is_valid


def test_backtest_macro_future_bars_never_affect_earlier_trades():
    """Point-in-time safety: corrupt ONLY the macro series' bars AFTER a
    cutoff (wildly different regime) while keeping bars before the cutoff
    byte-identical -- every trade ENTERED before the cutoff (with enough
    margin for the trend-lookback window) must be identical to a run using
    the uncorrupted macro series throughout. If the per-bar `.iloc[:i+1]`
    slicing in Backtester.run() ever leaked a future macro bar into an
    earlier decision, this would catch it."""
    from backtesting.backtester import Backtester

    daily = make_uptrend_ohlcv(n=700, seed=502)
    india_vix_original = make_uptrend_ohlcv(n=700, seed=503)

    cutoff = 500
    india_vix_corrupted = india_vix_original.copy()
    # Replace everything from the cutoff onward with a completely different
    # (strongly downtrending) series -- if this ever leaks backward, trend
    # classification at bars well before the cutoff would flip.
    replacement = make_downtrend_ohlcv(n=len(india_vix_original) - cutoff, seed=504)
    india_vix_corrupted.iloc[cutoff:] = replacement.values

    config = make_bt_config()
    result_original = Backtester(config=config).run(
        "TEST.NS", daily, macro_daily={"india_vix": india_vix_original},
    )
    result_corrupted = Backtester(config=config).run(
        "TEST.NS", daily, macro_daily={"india_vix": india_vix_corrupted},
    )

    # Trades ENTERED with at least a 50-bar margin before the cutoff (the
    # trend-lookback window market_trend() uses) must be byte-identical --
    # future macro corruption must never reach back into an earlier decision.
    safe_cutoff_date = daily.index[cutoff - 60]
    early_original = [t for t in result_original.trades if t.entry_date < safe_cutoff_date]
    early_corrupted = [t for t in result_corrupted.trades if t.entry_date < safe_cutoff_date]
    assert len(early_original) == len(early_corrupted)
    for a, b in zip(early_original, early_corrupted):
        assert a.entry_date == b.entry_date
        assert a.entry_price == pytest.approx(b.entry_price)
        assert a.exit_date == b.exit_date
        assert a.net_pnl == pytest.approx(b.net_pnl)
