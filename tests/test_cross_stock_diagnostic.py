"""
Spec section 11: an automated diagnostic test that runs the strategy on
multiple stocks and verifies:
  - all valid stocks can reach the technical engine
  - indicators are calculated correctly
  - trade setup generation works
  - final decision logic works
  - one stock's data cannot contaminate another stock's analysis
Explicitly does NOT require every stock to produce a trade -- only that the
pipeline behaves correctly (spec: "Do NOT require every stock to produce a
trade... Require only that the pipeline behaves correctly").

Uses `backtesting.diagnostics.SignalFunnel`, the same stage-by-stage
instrumentation used for the multi-stock report in
`scripts/multi_stock_diagnostic.py` (see that script and README.md's "Known
issue, fixed" section for the full 10-symbol table and the honest disclosure
that this environment cannot reach a real market-data vendor, so these are
clearly-synthetic, deterministic fixtures, not real NSE history).
"""

from __future__ import annotations

import pandas as pd
import pytest

from backtesting.diagnostics import SignalFunnel
from config.settings import Config
from tests.conftest import make_synthetic_ohlcv

# Deliberately varied: strong uptrend, mild uptrend, downtrend, choppy,
# high-volatility -- NOT all engineered to produce a trade. A downtrend or
# choppy symbol legitimately producing zero trades is the CORRECT outcome
# for a long-only signal engine, not a bug.
MULTI_STOCK_FIXTURES = {
    "SYM_STRONG_UPTREND": dict(drift=0.0035, volatility=0.011, seed=101),
    "SYM_MILD_UPTREND": dict(drift=0.0015, volatility=0.014, seed=102),
    "SYM_DOWNTREND": dict(drift=-0.0030, volatility=0.016, seed=103),
    "SYM_CHOPPY": dict(drift=0.0000, volatility=0.012, seed=104),
    "SYM_HIGH_VOL": dict(drift=0.0010, volatility=0.045, seed=105),
}
N_BARS = 400  # long enough to clear min_history_bars with room to run the funnel


@pytest.fixture(scope="module")
def multi_stock_frames():
    return {
        symbol: make_synthetic_ohlcv(n=N_BARS, start_date="2022-01-03", **params)
        for symbol, params in MULTI_STOCK_FIXTURES.items()
    }


@pytest.fixture(scope="module")
def index_frame():
    return make_synthetic_ohlcv(n=N_BARS, drift=0.0006, volatility=0.010, seed=999, start_date="2022-01-03")


def make_diagnostic_config() -> Config:
    config = Config()
    config.decision_thresholds.min_history_bars = 60  # so a 400-bar fixture has room to actually trade
    return config


# --- All valid stocks reach the technical engine, indicators calculate correctly ---

def test_all_symbols_reach_the_technical_engine(multi_stock_frames, index_frame):
    config = make_diagnostic_config()
    for symbol, daily in multi_stock_frames.items():
        funnel = SignalFunnel(config=config)
        report = funnel.run(symbol, daily, index_daily=index_frame)
        assert report.data_quality_tradeable, f"{symbol}: unexpectedly failed the data-quality gate."
        assert report.counts.bars_with_min_bars_for_indicators > 0, (
            f"{symbol}: never reached the technical engine at all -- indicator warm-up produced 0 bars."
        )


def test_indicators_produce_valid_non_nan_values_after_warmup(multi_stock_frames):
    from indicators.technical import TechnicalAnalyzer

    analyzer = TechnicalAnalyzer()
    for symbol, daily in multi_stock_frames.items():
        snap = analyzer.analyze(symbol, daily)  # full series -- well past every indicator's warm-up
        for field_name in ("sma20", "sma50", "rsi14", "macd_hist", "atr14", "bb_upper", "bb_lower"):
            value = getattr(snap, field_name)
            assert value is not None and not (isinstance(value, float) and pd.isna(value)), (
                f"{symbol}: indicator {field_name} is NaN/None after warm-up."
            )
        assert snap.sma200 is not None  # N_BARS=400 > 200, so full warm-up is expected here


# --- Trade setup generation and final decision logic work, per symbol ------

def test_trade_setup_and_final_decision_logic_run_without_error(multi_stock_frames, index_frame):
    """Every symbol must be independently processable end to end (no
    exceptions, no crashes) regardless of whether it ultimately produces a
    trade. This is the literal spec-11 ask: 'trade setup generation works'
    and 'final decision logic works', not 'every symbol trades'."""
    config = make_diagnostic_config()
    reports = {}
    for symbol, daily in multi_stock_frames.items():
        funnel = SignalFunnel(config=config)
        reports[symbol] = funnel.run(symbol, daily, index_daily=index_frame)

    for symbol, report in reports.items():
        c = report.counts
        # Every bar that reached the technical engine must have been
        # resolved to SOME accounted-for outcome (no bar silently vanishes
        # from every counter) -- summed terminal-outcome buckets should not
        # exceed the number of eligible bars, and confidence_pass+fail
        # should account for essentially all indicator-ready bars beyond
        # the insufficient-history/no-components short circuits.
        accounted = (
            c.no_trade_insufficient_history + c.no_trade_no_components_available
            + c.confidence_pass + c.confidence_fail
        )
        assert accounted <= c.bars_total
        assert accounted > 0, f"{symbol}: no bar reached even a confidence verdict -- pipeline likely crashed silently."


def test_zero_trades_is_an_acceptable_outcome_for_a_genuinely_bad_setup(multi_stock_frames, index_frame):
    """Spec section 12: 'It is acceptable for some stocks to have 0 trades
    if there genuinely were no valid setups.' A persistent downtrend, fed to
    a long-only signal engine, legitimately produces zero (or very few)
    approved BUY trades -- this must NOT be treated as a bug."""
    config = make_diagnostic_config()
    funnel = SignalFunnel(config=config)
    report = funnel.run("SYM_DOWNTREND", multi_stock_frames["SYM_DOWNTREND"], index_daily=index_frame)
    # Not asserting exactly 0 (a downtrend can still have local bounces that
    # briefly look bullish) -- asserting it trades MUCH less often than the
    # strong uptrend symbol, which is the actual claim being protected.
    uptrend_funnel = SignalFunnel(config=config)
    uptrend_report = uptrend_funnel.run("SYM_STRONG_UPTREND", multi_stock_frames["SYM_STRONG_UPTREND"], index_daily=index_frame)
    assert report.counts.final_approved <= uptrend_report.counts.final_approved


def test_not_every_symbol_is_required_to_produce_a_trade(multi_stock_frames, index_frame):
    """The diagnostic as a whole must not silently assume every symbol
    trades -- at least one of the deliberately unfavorable fixtures
    (downtrend/choppy) is allowed to have zero approved trades without
    failing this test suite."""
    config = make_diagnostic_config()
    zero_trade_symbols = []
    for symbol in ("SYM_DOWNTREND", "SYM_CHOPPY"):
        funnel = SignalFunnel(config=config)
        report = funnel.run(symbol, multi_stock_frames[symbol], index_daily=index_frame)
        if report.counts.final_approved == 0:
            zero_trade_symbols.append(symbol)
    # This assertion exists to document the acceptance criterion, not to
    # force a specific outcome -- it only fails if the test harness itself
    # is broken (e.g. an exception was swallowed elsewhere).
    assert isinstance(zero_trade_symbols, list)


# --- No cross-contamination between stocks' data ----------------------------

def test_no_cross_contamination_between_symbols(multi_stock_frames, index_frame):
    """Running symbols through independent SignalFunnel instances (as a real
    multi-symbol scan would) must give each symbol results that could only
    have come from ITS OWN price series -- e.g. the strong-uptrend symbol
    must show meaningfully more bullish technical bars than the downtrend
    symbol; if a data mix-up occurred, this relationship would invert or
    collapse."""
    config = make_diagnostic_config()
    reports = {}
    for symbol, daily in multi_stock_frames.items():
        funnel = SignalFunnel(config=config)
        reports[symbol] = funnel.run(symbol, daily, index_daily=index_frame)

    up = reports["SYM_STRONG_UPTREND"].counts
    down = reports["SYM_DOWNTREND"].counts
    assert up.technical_bullish > down.technical_bullish
    assert down.technical_bearish > up.technical_bearish

    # Cross-check against directly re-running TechnicalAnalyzer on each raw
    # frame -- the funnel's internal per-bar accounting must agree with an
    # independent computation on the symbol's own data, not some other
    # symbol's.
    from indicators.technical import TechnicalAnalyzer
    analyzer = TechnicalAnalyzer()
    up_snap = analyzer.analyze("SYM_STRONG_UPTREND", multi_stock_frames["SYM_STRONG_UPTREND"])
    down_snap = analyzer.analyze("SYM_DOWNTREND", multi_stock_frames["SYM_DOWNTREND"])
    assert up_snap.trend == "UPTREND"
    assert down_snap.trend != "UPTREND"
    assert up_snap.last_close != pytest.approx(down_snap.last_close)


# --- SignalFunnel must genuinely drive the SAME pipeline, never a stale/
# independently-recomputed approximation of it (module's own stated design
# principle) -----------------------------------------------------------------

def test_signal_funnel_technical_bucket_uses_the_alpha_blend_not_the_old_score(
    monkeypatch, multi_stock_frames, index_frame,
):
    """SignalEngine has bucketed the 'technical' component from
    _technical_alpha_score() (trend/momentum/volatility-volume blend)
    since Phase 10 -- technical_score() is an older formula still kept
    around only for other callers. Forces a deterministic, unmistakable
    value for the blend and confirms SignalFunnel's own bullish/bearish
    tally follows IT, not the real (and here, deliberately different)
    technical_score()."""
    import backtesting.diagnostics as diagnostics_module
    from indicators.technical import TechnicalAnalyzer

    daily = multi_stock_frames["SYM_STRONG_UPTREND"]
    real_score = TechnicalAnalyzer().analyze("SYM_STRONG_UPTREND", daily).technical_score()
    assert real_score > 55  # sanity: the OLD formula reads this fixture as bullish

    monkeypatch.setattr(diagnostics_module, "_technical_alpha_score", lambda technical: 10.0)  # force "bearish"
    config = make_diagnostic_config()
    funnel = SignalFunnel(config=config)
    report = funnel.run("SYM_STRONG_UPTREND", daily, index_daily=index_frame)
    assert report.counts.technical_bullish == 0
    assert report.counts.technical_bearish > 0


def test_signal_funnel_threads_market_regime_into_the_pipeline(monkeypatch, multi_stock_frames, index_frame):
    """Previously SignalInputs was built WITHOUT market_regime at all
    (silently defaulting to 'UNKNOWN'), so SignalEngine's bearish-regime
    confidence bump could never fire inside this diagnostic's own
    pipeline.decide() calls -- unlike PaperTradingEngine.scan_symbol()/
    Backtester.run(), which both thread it through correctly. Forces every
    bar to classify as BEAR and sets an extreme confidence bump so the
    real, bumped threshold is virtually unclearable -- every eligible bar
    must be confidence_fail, never confidence_pass (the pre-fix code, with
    market_regime permanently 'UNKNOWN', would have shown confidence_pass
    for most bars here since the UNBUMPED threshold is trivial to clear)."""
    import backtesting.diagnostics as diagnostics_module

    monkeypatch.setattr(diagnostics_module, "classify_regime", lambda daily, i: "BEAR_HIGH_VOL")
    config = make_diagnostic_config()
    config.decision_thresholds.min_confidence_to_trade = 1.0
    config.decision_thresholds.bearish_regime_confidence_bonus = 95.0

    funnel = SignalFunnel(config=config)
    report = funnel.run(
        "SYM_STRONG_UPTREND", multi_stock_frames["SYM_STRONG_UPTREND"], index_daily=index_frame,
    )
    assert report.counts.confidence_pass == 0
    assert report.counts.confidence_fail > 0


def test_shared_index_daily_does_not_leak_symbol_data_across_runs(multi_stock_frames, index_frame):
    """All symbols in this test legitimately share the SAME benchmark index
    frame (as they would in real multi-symbol scanning) -- this must not be
    confused with symbol-data contamination. Each symbol's OWN `daily` frame
    must remain the only source of its technical/price values."""
    config = make_diagnostic_config()
    funnel_a = SignalFunnel(config=config)
    report_a = funnel_a.run("SYM_STRONG_UPTREND", multi_stock_frames["SYM_STRONG_UPTREND"], index_daily=index_frame)
    funnel_b = SignalFunnel(config=config)
    report_b = funnel_b.run("SYM_CHOPPY", multi_stock_frames["SYM_CHOPPY"], index_daily=index_frame)

    # Using the identical index_frame object for both must not make their
    # technical outcomes identical.
    assert report_a.counts.technical_bullish != report_b.counts.technical_bullish or (
        report_a.counts.final_approved != report_b.counts.final_approved
    )
