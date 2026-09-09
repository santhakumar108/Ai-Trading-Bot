"""
Regression coverage for the "0 approved trades across every NSE symbol in a
5-year backtest" bug (see README.md's "Known issue, fixed" section).

Root cause: `config.decision_thresholds.min_independent_signals` defaulted
to 2, but a default `Backtester` (no fundamentals/news/social/ML provider
supplied) can only ever make `technical` available -- `NoHistorical*Provider`
always returns None for fundamentals/news/social, and `ml_provider` is None
unless explicitly wired in. `available_independent_signals` was therefore
ALWAYS exactly 1, so `strategy/signal_engine.py`'s
`available_independent_signals < min_independent_signals` gate fired
unconditionally on every bar of every symbol -- before market regime,
confidence, R:R, or expected value were ever meaningfully exercised. This
was a structural config/architecture mismatch, not "genuinely no valid
setups", and not a broken indicator/R:R/EV calculation.

These tests pin the fix (default is now 1) and prove a genuine technical-only
bullish setup CAN pass through the complete, UNMODIFIED default pipeline
end-to-end -- satisfying spec section 12's acceptance criterion: "The system
must demonstrate that genuine historical setups can pass through the
pipeline when all required conditions are actually satisfied."
"""

from __future__ import annotations

import pandas as pd
import pytest

from backtesting.backtester import Backtester, BacktestConfig
from config.settings import Config, DecisionThresholds, load_config
from strategy.pipeline import build_pipeline
from strategy.signal_engine import SignalEngine, SignalInputs
from tests.conftest import make_synthetic_ohlcv


# --- Pin the fixed default --------------------------------------------------

def test_default_min_independent_signals_is_one_not_two():
    """The dataclass default (used by Config() directly, e.g. in most unit
    tests and anywhere a Config is built in code) must be 1."""
    assert DecisionThresholds().min_independent_signals == 1
    assert Config().decision_thresholds.min_independent_signals == 1


def test_shipped_yaml_default_min_independent_signals_is_one():
    """The actual shipped config/default_config.yaml (what a real `python
    main.py` run or a real Backtester(load_config()) call actually uses)
    must match the dataclass default -- this is the file a real deployment
    reads, not just the in-code dataclass fallback."""
    config = load_config(override_path="__nonexistent_override__.yaml")
    assert config.decision_thresholds.min_independent_signals == 1


# --- Reproduce the bug mechanically (would fail before the fix) ------------

def test_technical_only_signal_is_never_enough_when_gate_set_to_two():
    """Sanity check that the GATE ITSELF still works correctly when an
    operator explicitly wants stricter multi-signal corroboration (e.g. once
    real fundamentals/news/social are wired in) -- lowering the DEFAULT must
    not have removed the mechanism, only recalibrated its out-of-the-box
    value. This mirrors tests/test_signal_engine.py's
    test_min_independent_signals_gate_blocks_single_signal_trades."""
    from tests.test_signal_engine import make_engine, base_inputs

    engine = make_engine(min_independent_signals=2)
    inputs = base_inputs(fundamentals=None, news=None, social=None)
    decision = engine.decide(inputs)
    assert decision.available_independent_signals == 1
    assert decision.decision == "NO TRADE"


def test_default_backtest_with_old_buggy_threshold_produces_zero_trades_on_every_symbol():
    """Demonstrates the ORIGINAL bug directly: with min_independent_signals
    explicitly restored to the old default (2), a default Backtester (no
    fundamentals/news/social/ML provider wired in) produces ZERO approved
    trades on MULTIPLE independent symbols with genuinely different price
    behavior (strong uptrend, downtrend, choppy) -- proving the zero-trade
    outcome was a common, symbol-independent gate, not "no valid setups" on
    any of them individually."""
    symbols = {
        "UPTREND": make_synthetic_ohlcv(n=400, drift=0.004, volatility=0.010, seed=11),
        "DOWNTREND": make_synthetic_ohlcv(n=400, drift=-0.004, volatility=0.010, seed=12),
        "CHOPPY": make_synthetic_ohlcv(n=400, drift=0.0, volatility=0.010, seed=13),
    }
    for name, daily in symbols.items():
        config = Config()
        config.decision_thresholds.min_history_bars = 60
        config.decision_thresholds.min_independent_signals = 2  # the OLD, buggy default
        bt = Backtester(config=config)
        result = bt.run(name, daily)
        assert result.signals_approved == 0, (
            f"{name}: expected the old min_independent_signals=2 default to block every trade, "
            f"but {result.signals_approved} were approved."
        )
        # Confirm it really is the independent-signals gate doing the blocking, not something
        # else about this particular symbol -- every rejected bar's SignalEngine call should be
        # capable of reaching the gate at all (i.e. it is not the data-quality gate that's
        # short-circuiting everything before SignalEngine even runs).
        assert result.is_valid is True
        assert result.data_quality.is_tradeable() is True


# --- Prove the fix: a genuine setup CAN pass end-to-end with the fixed default ----

def test_default_config_backtest_approves_at_least_one_genuine_setup():
    """The core acceptance criterion (spec section 12): using the ACTUAL,
    UNMODIFIED default config (loaded via load_config(), not a hand-tuned
    test config), a long, strong, low-noise uptrend with ample history
    (>= min_history_bars) and a real benchmark index series must be able to
    produce at least one approved BUY trade. If this fails, either the
    independent-signals fix regressed, the market-condition-availability fix
    regressed, or some other gate (confidence, R:R, EV, data quality) is
    newly and unconditionally blocking every trade -- all are things this
    regression test exists to catch."""
    config = load_config(override_path="__nonexistent_override__.yaml")
    daily = make_synthetic_ohlcv(n=1250, drift=0.0035, volatility=0.011, seed=1001, start_date="2021-01-04")
    index_daily = make_synthetic_ohlcv(n=1250, drift=0.0008, volatility=0.012, seed=9001, start_date="2021-01-04")
    bt = Backtester(config=config, backtest_config=BacktestConfig(initial_capital=1_000_000.0))
    result = bt.run("DIAGNOSTIC_STRONG_UPTREND", daily, index_daily=index_daily)

    assert result.is_valid, "Run-level data-quality gate unexpectedly failed for clean synthetic data."
    assert result.signals_approved >= 1, (
        "Expected at least one approved trade for a strong, long, low-noise uptrend under the "
        "UNMODIFIED default config -- if this is 0, a gate is unconditionally blocking every "
        "trade again (this is exactly the bug this test suite guards against)."
    )
    assert len(result.trades) >= 1
    trade = result.trades[0]
    # Every approved trade must still satisfy every real risk control -- the
    # fix must not have weakened these.
    assert trade.side == "BUY"
    reward = trade.target - trade.entry_price if False else None  # not directly on Trade; check via risk fields instead
    assert "fundamentals" in trade.unavailable_components  # honestly excluded, never fabricated
    assert "news_sentiment" in trade.unavailable_components
    assert "social_sentiment" in trade.unavailable_components
    assert trade.confidence_at_entry >= config.decision_thresholds.min_confidence_to_trade


def test_default_config_backtest_still_enforces_minimum_risk_reward():
    """The fix must not have loosened the 2:1 minimum R:R floor -- every
    approved trade's proposed stop/target (computed by
    StrategyPipeline.propose_stop_target, unchanged by this fix) must still
    clear it before transaction costs, by construction."""
    config = load_config(override_path="__nonexistent_override__.yaml")
    assert config.decision_thresholds.min_risk_reward == pytest.approx(2.0)
    daily = make_synthetic_ohlcv(n=1250, drift=0.0035, volatility=0.011, seed=1001, start_date="2021-01-04")
    index_daily = make_synthetic_ohlcv(n=1250, drift=0.0008, volatility=0.012, seed=9001, start_date="2021-01-04")
    bt = Backtester(config=config)
    result = bt.run("DIAGNOSTIC_STRONG_UPTREND", daily, index_daily=index_daily)
    for trade in result.trades:
        # Reconstruct the pre-cost R:R from the recorded absolute levels.
        risk_per_share = trade.entry_price - trade.stop_loss
        reward_per_share = trade.target - trade.entry_price
        assert risk_per_share > 0
        assert reward_per_share / risk_per_share >= 2.0 - 1e-9


def test_default_config_backtest_still_enforces_expected_value_gate():
    """The fix must not bypass the expected-value gate -- every approved
    trade must have had a non-negative expected value at approval time
    (spec section 10: 'do not optimize only for win rate')."""
    config = load_config(override_path="__nonexistent_override__.yaml")
    pipeline = build_pipeline(config)
    daily = make_synthetic_ohlcv(n=1250, drift=0.0035, volatility=0.011, seed=1001, start_date="2021-01-04")
    index_daily = make_synthetic_ohlcv(n=1250, drift=0.0008, volatility=0.012, seed=9001, start_date="2021-01-04")
    bt = Backtester(config=config, pipeline=pipeline)
    result = bt.run("DIAGNOSTIC_STRONG_UPTREND", daily, index_daily=index_daily)
    assert result.signals_approved >= 1
    # Re-evaluate the risk assessment for the first trade's exact entry/stop/target
    # through the SAME risk calculator the pipeline used, to confirm EV was positive.
    trade = result.trades[0]
    risk = pipeline.risk_calculator.evaluate(
        symbol="DIAGNOSTIC_STRONG_UPTREND", side="BUY", entry=trade.entry_price,
        stop_loss=trade.stop_loss, target=trade.target, capital=1_000_000.0,
    )
    assert risk.expected_value_positive is True


# --- Second discovered root cause: missing benchmark data silently gated every trade ---

def test_missing_benchmark_index_data_no_longer_blocks_every_trade():
    """A SECOND structural bug was found during this investigation, in the
    same family as the min_independent_signals one: `strategy/trade_filter.py`
    computed `checklist["market_conditions_acceptable"] =
    component_scores.get("market_condition", 0) >= 40`. When no benchmark
    index data is available (no `index_daily` passed to `Backtester.run()`,
    no `universe.index_symbol` configured, or the index fetch failed),
    `strategy/signal_engine.py` correctly EXCLUDES `market_condition` from
    `component_scores` (never fabricates a neutral/positive score) -- but
    the absent dict key then silently read back as 0 via `.get(..., 0)` and
    failed this check UNCONDITIONALLY, turning "we don't have benchmark
    data" into an undisclosed mandatory gate that blocked every trade. This
    test proves a `Backtester.run()` call with NO `index_daily` supplied at
    all (a completely realistic call pattern -- several tests in
    test_backtester.py call `bt.run("TEST", uptrend_daily)` this exact way)
    can still approve a genuine setup after the fix (see
    strategy/trade_filter.py's `market_condition_available` parameter and
    strategy/pipeline.py's `decide()`, which now passes it through)."""
    config = load_config(override_path="__nonexistent_override__.yaml")
    daily = make_synthetic_ohlcv(n=1250, drift=0.0035, volatility=0.011, seed=1001, start_date="2021-01-04")
    bt = Backtester(config=config, backtest_config=BacktestConfig(initial_capital=1_000_000.0))
    result = bt.run("DIAGNOSTIC_NO_BENCHMARK", daily)  # deliberately no index_daily=

    assert result.is_valid
    assert result.signals_approved >= 1, (
        "Expected at least one approved trade for a strong uptrend even with NO benchmark index "
        "data supplied -- if this is 0, the market_condition-unavailable gate is blocking every "
        "trade again."
    )
    for trade in result.trades:
        assert "market_condition" in trade.unavailable_components
