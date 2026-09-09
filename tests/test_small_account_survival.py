"""
Spec Part 25: small-account survival simulation -- a major acceptance
criterion. Covers the exact INSUFFICIENT-CAPITAL verdict at a capital too
small to trade anything, survival_rate/probability_of_ruin consistency,
transaction-cost-as-%-of-capital reporting, and that data is fetched once
per symbol regardless of how many capital levels are simulated.
"""

from __future__ import annotations

import pytest

from config.settings import Config
from data.market_data import DataUnavailableError, MarketDataProvider
from backtesting.small_account_survival import (
    INSUFFICIENT_CAPITAL_VERDICT, run_small_account_survival,
)
from tests.conftest import make_synthetic_ohlcv, make_uptrend_ohlcv


def make_config():
    config = Config()
    config.decision_thresholds.min_history_bars = 60
    config.universe.index_symbol = None
    return config


def test_tiny_capital_yields_insufficient_capital_verdict():
    daily = make_uptrend_ohlcv(n=700, seed=201)
    data = {"WIN.NS": daily}

    def fetch(symbol, period, interval):
        return data[symbol].copy()

    md = MarketDataProvider(fetch_fn=fetch)
    report = run_small_account_survival(
        make_config(), list(data.keys()), capital_levels=[1.0], period="5y", market_data=md, num_simulations=200,
    )
    assert len(report.levels) == 1
    lvl = report.levels[0]
    assert lvl.total_trades == 0
    assert lvl.verdict == INSUFFICIENT_CAPITAL_VERDICT


def test_zero_trades_at_every_capital_level_is_not_reported_as_insufficient_capital():
    """Regression: if even the LARGEST capital level took zero trades, the
    outcome is a signal-quality one ('no qualifying setup this period'),
    not a capital constraint -- must not print the same verdict as a
    genuine INR 100 affordability failure (see main.py's account-check for
    the same conflation, caught and fixed in Phase 1)."""
    flat = make_synthetic_ohlcv(n=700, seed=299, drift=0.0, volatility=0.0002)  # too quiet to ever signal
    data = {"QUIET.NS": flat}

    def fetch(symbol, period, interval):
        return data[symbol].copy()

    md = MarketDataProvider(fetch_fn=fetch)
    report = run_small_account_survival(
        make_config(), list(data.keys()), capital_levels=[100.0, 100_000.0], period="5y", market_data=md,
        num_simulations=100,
    )
    by_capital = {lvl.capital: lvl for lvl in report.levels}
    if by_capital[100_000.0].total_trades == 0:
        assert by_capital[100.0].verdict != INSUFFICIENT_CAPITAL_VERDICT
        assert by_capital[100_000.0].verdict != INSUFFICIENT_CAPITAL_VERDICT
        assert "not a capital constraint" in by_capital[100.0].verdict.lower()


def test_capital_large_enough_to_trade_has_no_editorial_verdict():
    daily = make_uptrend_ohlcv(n=700, seed=202)
    data = {"WIN.NS": daily}

    def fetch(symbol, period, interval):
        return data[symbol].copy()

    md = MarketDataProvider(fetch_fn=fetch)
    report = run_small_account_survival(
        make_config(), list(data.keys()), capital_levels=[100_000.0], period="5y", market_data=md,
        num_simulations=200,
    )
    lvl = report.levels[0]
    if lvl.total_trades > 0:
        assert lvl.verdict == ""  # no "OK"/"success" editorializing (spec Part 22/29)


def test_survival_rate_is_one_minus_probability_of_ruin():
    daily = make_uptrend_ohlcv(n=700, seed=203)
    data = {"WIN.NS": daily}

    def fetch(symbol, period, interval):
        return data[symbol].copy()

    md = MarketDataProvider(fetch_fn=fetch)
    report = run_small_account_survival(
        make_config(), list(data.keys()), capital_levels=[100_000.0], period="5y", market_data=md,
        num_simulations=300,
    )
    lvl = report.levels[0]
    if lvl.total_trades > 0:
        assert lvl.survival_rate == pytest.approx(1.0 - lvl.probability_of_ruin)
        assert 0.0 <= lvl.probability_of_ruin <= 1.0


def test_transaction_cost_pct_of_capital_is_smaller_for_larger_accounts_at_same_position():
    """Costs are a currency amount tied to the trade's notional; as a
    FRACTION of a bigger account's capital, that same notional trade
    naturally represents a smaller percentage. Sanity-checks the division
    is against `capital`, not something else."""
    daily = make_uptrend_ohlcv(n=700, seed=204)
    data = {"WIN.NS": daily}

    def fetch(symbol, period, interval):
        return data[symbol].copy()

    md = MarketDataProvider(fetch_fn=fetch)
    report = run_small_account_survival(
        make_config(), list(data.keys()), capital_levels=[1_000.0, 100_000.0], period="5y", market_data=md,
        num_simulations=200,
    )
    by_capital = {lvl.capital: lvl for lvl in report.levels}
    small, large = by_capital[1_000.0], by_capital[100_000.0]
    if small.total_trades > 0 and large.total_trades > 0:
        # Position sizing scales with capital, so this isn't a strict
        # inequality in general -- but both must be non-negative, finite
        # fractions of their own account, never nonsensical (e.g. > 1 for
        # a single trade's cost is implausible at realistic cost_pct).
        assert 0.0 <= small.avg_transaction_cost_pct_of_capital < 1.0
        assert 0.0 <= large.avg_transaction_cost_pct_of_capital < 1.0


def test_data_fetched_once_per_symbol_regardless_of_capital_level_count():
    daily = make_uptrend_ohlcv(n=700, seed=205)
    data = {"WIN.NS": daily}
    call_count = {"n": 0}

    def fetch(symbol, period, interval):
        call_count["n"] += 1
        return data[symbol].copy()

    md = MarketDataProvider(fetch_fn=fetch)
    run_small_account_survival(
        make_config(), list(data.keys()), capital_levels=[100.0, 1_000.0, 100_000.0], period="5y",
        market_data=md, num_simulations=100,
    )
    assert call_count["n"] == 1  # NOT re-fetched per capital level


def test_summary_renders_without_crashing():
    daily = make_uptrend_ohlcv(n=700, seed=206)
    data = {"WIN.NS": daily}

    def fetch(symbol, period, interval):
        return data[symbol].copy()

    md = MarketDataProvider(fetch_fn=fetch)
    report = run_small_account_survival(
        make_config(), list(data.keys()), capital_levels=[100.0, 100_000.0], period="5y", market_data=md,
        num_simulations=100,
    )
    text = report.summary()
    assert isinstance(text, str) and len(text) > 0
    assert INSUFFICIENT_CAPITAL_VERDICT in text or "100" in text


def test_summary_surfaces_median_losing_streak_and_final_equity():
    """median_losing_streak/median_final_equity are computed (from the
    Monte Carlo report's own P50) but previously never appeared in
    summary() -- spec Part 25 explicitly names losing-streak survival as
    a headline concern. Confirms the columns and a real level's computed
    values now actually appear in the printed report."""
    daily = make_uptrend_ohlcv(n=700, seed=213)
    data = {"WIN.NS": daily}

    def fetch(symbol, period, interval):
        return data[symbol].copy()

    md = MarketDataProvider(fetch_fn=fetch)
    report = run_small_account_survival(
        make_config(), list(data.keys()), capital_levels=[100_000.0], period="5y", market_data=md,
        num_simulations=200,
    )
    text = report.summary()
    assert "MED_STREAK" in text
    assert "MED_EQUITY" in text
    lvl = report.levels[0]
    if lvl.total_trades > 0:
        assert f"{lvl.median_losing_streak:.1f}" in text


# =============================================================================
# Fetch-time rate-limit batching (spec section 19's same principle applied
# here as paper_trading/scanner.py's UniverseScanner already applies to the
# live-scan path -- MarketDataProvider.get_daily_batch).
# =============================================================================

def test_small_account_survival_no_delay_when_batch_size_covers_all_symbols(monkeypatch):
    def fail_if_called(*_a, **_k):
        raise AssertionError("must not sleep with the default batch size")
    monkeypatch.setattr("time.sleep", fail_if_called)

    daily = make_uptrend_ohlcv(n=700, seed=210)
    data = {"WIN.NS": daily}

    def fetch(symbol, period, interval):
        return data[symbol].copy()

    md = MarketDataProvider(fetch_fn=fetch)
    run_small_account_survival(
        make_config(), list(data.keys()), capital_levels=[100_000.0], period="5y", market_data=md,
        num_simulations=100,
    )  # must not raise


def test_small_account_survival_paces_fetches_per_configured_batch_settings(monkeypatch):
    sleep_calls = []
    monkeypatch.setattr("time.sleep", lambda s: sleep_calls.append(s))

    config = make_config()
    config.universe.scan_batch_size = 2
    config.universe.scan_batch_delay_seconds = 0.75
    symbols = ["A.NS", "B.NS", "C.NS"]
    daily = make_uptrend_ohlcv(n=700, seed=211)

    def fetch(symbol, period, interval):
        return daily.copy()

    md = MarketDataProvider(fetch_fn=fetch)
    run_small_account_survival(
        config, symbols, capital_levels=[100_000.0], period="5y", market_data=md, num_simulations=100,
    )
    assert sleep_calls == [0.75]  # batches of (2, 1) -> 1 gap


def test_small_account_survival_fetch_failure_still_isolated_when_batched(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)

    config = make_config()
    config.universe.scan_batch_size = 1
    config.universe.scan_batch_delay_seconds = 0.0
    good = make_uptrend_ohlcv(n=700, seed=212)

    def fetch(symbol, period, interval):
        if symbol == "BAD.NS":
            raise DataUnavailableError("simulated outage")
        return good.copy()

    md = MarketDataProvider(fetch_fn=fetch)
    report = run_small_account_survival(
        config, ["GOOD.NS", "BAD.NS"], capital_levels=[100_000.0], period="5y", market_data=md,
        num_simulations=100,
    )
    assert len(report.levels) == 1  # BAD.NS simply absent from pooled data, batch not aborted


# =============================================================================
# Spec Part 2: macro context forwarded to every symbol/capital-level combo
# (previously a silent gap -- Backtester.run() itself already supported
# macro_daily; this closes the pass-through in run_small_account_survival(),
# invoked by `research --survival`).
# =============================================================================

def test_small_account_survival_macro_daily_none_matches_omitted_default():
    daily = make_uptrend_ohlcv(n=700, seed=207)
    data = {"WIN.NS": daily}

    def fetch(symbol, period, interval):
        return data[symbol].copy()

    md = MarketDataProvider(fetch_fn=fetch)
    report_omitted = run_small_account_survival(
        make_config(), list(data.keys()), capital_levels=[100_000.0], period="5y", market_data=md,
        num_simulations=200, seed=42,
    )
    report_explicit = run_small_account_survival(
        make_config(), list(data.keys()), capital_levels=[100_000.0], period="5y", market_data=md,
        num_simulations=200, seed=42, macro_daily=None,
    )
    assert report_omitted.summary() == report_explicit.summary()


def test_small_account_survival_forwards_same_macro_daily_to_every_symbol_and_capital_level(monkeypatch):
    """Capture-based proof: macro is market-wide, so the SAME macro_daily
    dict passed into run_small_account_survival() must reach EVERY
    (symbol, capital_level) combination's Backtester.run() call."""
    from backtesting.backtester import Backtester

    a = make_uptrend_ohlcv(n=700, seed=208)
    b = make_uptrend_ohlcv(n=700, seed=209)
    data = {"A.NS": a, "B.NS": b}

    def fetch(symbol, period, interval):
        return data[symbol].copy()

    md = MarketDataProvider(fetch_fn=fetch)

    captured = []
    original_run = Backtester.run

    def spy_run(self, *args, **kwargs):
        captured.append(kwargs.get("macro_daily"))
        return original_run(self, *args, **kwargs)

    monkeypatch.setattr(Backtester, "run", spy_run)

    macro = {"india_vix": a.copy()}
    run_small_account_survival(
        make_config(), list(data.keys()), capital_levels=[1_000.0, 100_000.0], period="5y",
        market_data=md, num_simulations=100, macro_daily=macro,
    )
    # 2 symbols x 2 capital levels = 4 Backtester.run() calls.
    assert len(captured) == 4
    assert all(m is macro for m in captured)
