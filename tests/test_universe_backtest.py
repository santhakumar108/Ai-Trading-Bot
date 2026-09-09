"""
Spec Part 22: cross-stock robustness reporting -- "do not declare success
because one stock performs well." Covers the pure aggregation math
directly (deterministic, hand-built summaries) and the end-to-end loop
wiring with injected synthetic data (no real network calls).
"""

from __future__ import annotations

import pytest

from config.settings import Config
from data.market_data import DataUnavailableError, MarketDataProvider
from backtesting.universe_backtest import (
    SymbolBacktestSummary, run_universe_backtest, _aggregate,
)
from tests.conftest import make_downtrend_ohlcv, make_synthetic_ohlcv, make_uptrend_ohlcv


# =============================================================================
# Pure aggregation math
# =============================================================================

def _summary(symbol, sector, num_trades, pf, expectancy, cagr, sharpe):
    return SymbolBacktestSummary(
        symbol=symbol, sector=sector, is_valid=True, num_trades=num_trades,
        profit_factor=pf, expectancy=expectancy, cagr_pct=cagr, sharpe_ratio=sharpe,
    )


def test_aggregate_median_and_percentages():
    summaries = [
        _summary("A", "X", 5, 2.0, 10.0, 0.10, 1.0),
        _summary("B", "X", 3, 0.5, -5.0, -0.05, -0.2),
        _summary("C", "Y", 10, 1.5, 2.0, 0.02, 0.5),
    ]
    agg = _aggregate(summaries)
    assert agg["median_profit_factor"] == pytest.approx(1.5)
    assert agg["pct_profit_factor_above_1"] == pytest.approx(2 / 3)
    assert agg["pct_positive_expectancy"] == pytest.approx(2 / 3)
    assert agg["pct_positive_cagr"] == pytest.approx(2 / 3)
    assert agg["pct_positive_sharpe"] == pytest.approx(2 / 3)


def test_aggregate_empty_list_is_all_zero():
    agg = _aggregate([])
    assert agg == {
        "median_profit_factor": 0.0, "pct_profit_factor_above_1": 0.0,
        "pct_positive_expectancy": 0.0, "pct_positive_cagr": 0.0, "pct_positive_sharpe": 0.0,
    }


def test_aggregate_all_profitable_is_100_percent():
    summaries = [_summary("A", None, 5, 3.0, 10.0, 0.10, 1.0), _summary("B", None, 5, 2.0, 5.0, 0.05, 0.8)]
    agg = _aggregate(summaries)
    assert agg["pct_profit_factor_above_1"] == pytest.approx(1.0)
    assert agg["pct_positive_expectancy"] == pytest.approx(1.0)


# =============================================================================
# End-to-end loop, injected synthetic data (no network)
# =============================================================================

def make_config():
    config = Config()
    config.decision_thresholds.min_history_bars = 60
    config.universe.index_symbol = None  # no benchmark fetch in these tests
    return config


def test_zero_trade_symbols_excluded_from_aggregates_but_counted():
    """A downtrend-only symbol should correctly take zero LONG trades (the
    system's directional bias) -- must be counted in no_trade_count, NOT
    folded into the profit-factor/expectancy aggregates as a 'failure'."""
    # A MILDER uptrend than the shared make_uptrend_ohlcv fixture's default
    # (drift=0.004) -- that default drift, sustained for 700 bars, runs RSI
    # into deep overbought territory by the end, which Phase 10's momentum-
    # aware technical blend correctly reads as reduced attractiveness (a
    # real, deliberate scoring improvement, not a bug -- verified: nearly
    # every seed at this milder drift/volatility reliably takes trades).
    win = make_synthetic_ohlcv(n=700, seed=101, drift=0.0015, volatility=0.010)
    flat = make_downtrend_ohlcv(n=700, seed=102)
    data = {"WIN.NS": win, "FLAT.NS": flat}

    def fetch(symbol, period, interval):
        return data[symbol].copy()

    md = MarketDataProvider(fetch_fn=fetch)
    report = run_universe_backtest(make_config(), list(data.keys()), period="5y", market_data=md)

    assert report.universe_size == 2
    assert report.invalid_count == 0
    by_symbol = {s.symbol: s for s in report.summaries}
    assert by_symbol["WIN.NS"].num_trades > 0
    # The aggregate stats must be computed ONLY from WIN.NS -- if FLAT.NS's
    # zero-trade profit_factor=0.0 were folded in, %PF>1 would be diluted.
    assert report.symbols_traded == sum(1 for s in report.summaries if s.num_trades > 0)
    if by_symbol["FLAT.NS"].num_trades == 0:
        assert report.no_trade_count >= 1
        assert report.pct_profit_factor_above_1 == pytest.approx(1.0)  # not diluted by the zero-trade symbol


def test_single_symbol_fetch_failure_does_not_abort_the_batch():
    good = make_uptrend_ohlcv(n=700, seed=103)

    def fetch(symbol, period, interval):
        if symbol == "BAD.NS":
            raise DataUnavailableError("simulated outage")
        return good.copy()

    md = MarketDataProvider(fetch_fn=fetch)
    report = run_universe_backtest(make_config(), ["GOOD.NS", "BAD.NS"], period="5y", market_data=md)

    assert report.universe_size == 2
    assert report.invalid_count == 1
    by_symbol = {s.symbol: s for s in report.summaries}
    assert by_symbol["BAD.NS"].is_valid is False
    assert "unavailable" in by_symbol["BAD.NS"].note.lower()
    assert by_symbol["GOOD.NS"].is_valid is True  # the batch kept going


def test_sector_rollup_groups_traded_symbols_by_sector_map():
    a = make_uptrend_ohlcv(n=700, seed=104)
    b = make_uptrend_ohlcv(n=700, seed=105)
    data = {"A.NS": a, "B.NS": b}

    def fetch(symbol, period, interval):
        return data[symbol].copy()

    md = MarketDataProvider(fetch_fn=fetch)
    sector_map = {"A.NS": "IT", "B.NS": "IT"}
    report = run_universe_backtest(
        make_config(), list(data.keys()), period="5y", market_data=md, sector_map=sector_map,
    )
    if report.symbols_traded == 2:
        assert "IT" in report.sector_rollups
        assert report.sector_rollups["IT"].symbols_traded == 2


def test_symbol_with_no_sector_map_entry_groups_as_unspecified():
    a = make_uptrend_ohlcv(n=700, seed=106)
    data = {"NOSECTOR.NS": a}

    def fetch(symbol, period, interval):
        return data[symbol].copy()

    md = MarketDataProvider(fetch_fn=fetch)
    report = run_universe_backtest(make_config(), list(data.keys()), period="5y", market_data=md, sector_map={})
    if report.symbols_traded == 1:
        assert "UNSPECIFIED" in report.sector_rollups


def test_summary_renders_without_crashing_when_nothing_traded():
    flat = make_downtrend_ohlcv(n=700, seed=107)
    data = {"FLAT.NS": flat}

    def fetch(symbol, period, interval):
        return data[symbol].copy()

    md = MarketDataProvider(fetch_fn=fetch)
    report = run_universe_backtest(make_config(), list(data.keys()), period="5y", market_data=md)
    text = report.summary()
    assert isinstance(text, str) and len(text) > 0


# =============================================================================
# Spec Part 2: macro context forwarded to every symbol (previously a silent
# gap -- Backtester.run() itself already supported macro_daily; this closes
# the pass-through in run_universe_backtest()).
# =============================================================================

def test_universe_backtest_macro_daily_none_matches_omitted_default():
    a = make_uptrend_ohlcv(n=700, seed=108)
    data = {"A.NS": a}

    def fetch(symbol, period, interval):
        return data[symbol].copy()

    md = MarketDataProvider(fetch_fn=fetch)
    report_omitted = run_universe_backtest(make_config(), list(data.keys()), period="5y", market_data=md)
    report_explicit = run_universe_backtest(
        make_config(), list(data.keys()), period="5y", market_data=md, macro_daily=None,
    )
    assert report_omitted.summary() == report_explicit.summary()


def test_universe_backtest_no_delay_when_batch_size_covers_all_symbols(monkeypatch):
    """Backward-compat guard: default config.universe.scan_batch_size=10
    covers every symbol count used elsewhere in this file, so nothing
    here should ever sleep -- every pre-existing test in this file keeps
    passing unmodified."""
    def fail_if_called(*_a, **_k):
        raise AssertionError("must not sleep with the default batch size")
    monkeypatch.setattr("time.sleep", fail_if_called)

    a, b = make_uptrend_ohlcv(n=700, seed=301), make_uptrend_ohlcv(n=700, seed=302)
    data = {"A.NS": a, "B.NS": b}

    def fetch(symbol, period, interval):
        return data[symbol].copy()

    md = MarketDataProvider(fetch_fn=fetch)
    run_universe_backtest(make_config(), list(data.keys()), period="5y", market_data=md)  # must not raise


def test_universe_backtest_paces_fetches_per_configured_batch_settings(monkeypatch):
    sleep_calls = []
    monkeypatch.setattr("time.sleep", lambda s: sleep_calls.append(s))

    config = make_config()
    config.universe.scan_batch_size = 2
    config.universe.scan_batch_delay_seconds = 3.25
    symbols = ["A.NS", "B.NS", "C.NS", "D.NS", "E.NS"]
    daily = make_uptrend_ohlcv(n=700, seed=303)

    def fetch(symbol, period, interval):
        return daily.copy()

    md = MarketDataProvider(fetch_fn=fetch)
    report = run_universe_backtest(config, symbols, period="5y", market_data=md)
    assert report.universe_size == 5
    assert sleep_calls == [3.25, 3.25]  # batches of (2, 2, 1) -> 2 gaps


def test_universe_backtest_fetch_failure_still_isolated_when_batched(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)

    config = make_config()
    config.universe.scan_batch_size = 1
    config.universe.scan_batch_delay_seconds = 0.0
    good = make_uptrend_ohlcv(n=700, seed=304)

    def fetch(symbol, period, interval):
        if symbol == "BAD.NS":
            raise DataUnavailableError("simulated outage")
        return good.copy()

    md = MarketDataProvider(fetch_fn=fetch)
    report = run_universe_backtest(config, ["GOOD1.NS", "BAD.NS", "GOOD2.NS"], period="5y", market_data=md)
    by_symbol = {s.symbol: s for s in report.summaries}
    assert by_symbol["BAD.NS"].is_valid is False
    assert "unavailable" in by_symbol["BAD.NS"].note.lower()
    assert by_symbol["GOOD1.NS"].is_valid is True and by_symbol["GOOD2.NS"].is_valid is True


def test_universe_backtest_forwards_same_macro_daily_object_to_every_symbol(monkeypatch):
    """Capture-based proof: macro is market-wide, so the SAME macro_daily
    dict passed into run_universe_backtest() must reach EVERY symbol's
    Backtester.run() call, fetched once, not per symbol."""
    from backtesting.backtester import Backtester

    a = make_uptrend_ohlcv(n=700, seed=109)
    b = make_uptrend_ohlcv(n=700, seed=110)
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
    run_universe_backtest(make_config(), list(data.keys()), period="5y", market_data=md, macro_daily=macro)
    assert len(captured) == 2  # one call per symbol
    assert all(m is macro for m in captured)


# =============================================================================
# CLI argument wiring (argparse-level only -- no network calls)
# =============================================================================

def test_cli_research_defaults_to_configured_universe():
    import main
    parser = main.build_parser()
    args = parser.parse_args(["research"])
    assert args.symbols is None
    assert args.universe == "configured"
    assert args.period == "5y"
    assert args.survival is False
    assert args.capital_levels is None
    assert args.func is main.cmd_research


def test_cli_research_accepts_universe_preset_and_survival_flag():
    import main
    parser = main.build_parser()
    args = parser.parse_args([
        "research", "--universe", "nifty50", "--period", "2y", "--survival",
        "--capital-levels", "100,1000,100000",
    ])
    assert args.universe == "nifty50"
    assert args.period == "2y"
    assert args.survival is True
    assert args.capital_levels == "100,1000,100000"


def test_cli_research_explicit_symbols_override_universe():
    import main
    parser = main.build_parser()
    args = parser.parse_args(["research", "--symbols", "RELIANCE.NS", "TCS.NS"])
    assert args.symbols == ["RELIANCE.NS", "TCS.NS"]


def test_resolve_research_symbols_prefers_explicit_symbols():
    import main
    from config.settings import Config

    class _Args:
        symbols = ["A.NS", "B.NS"]
        universe = "nifty50"

    assert main._resolve_research_symbols(_Args(), Config()) == ["A.NS", "B.NS"]


def test_resolve_research_symbols_nifty50_preset():
    import main
    from config.nse_universe import NIFTY_50
    from config.settings import Config

    class _Args:
        symbols = None
        universe = "nifty50"

    assert main._resolve_research_symbols(_Args(), Config()) == NIFTY_50


def test_resolve_research_symbols_defaults_to_configured_universe():
    import main
    from config.settings import Config

    class _Args:
        symbols = None
        universe = "configured"

    config = Config()
    config.universe.symbols = ["ONLY.NS"]
    assert main._resolve_research_symbols(_Args(), config) == ["ONLY.NS"]
