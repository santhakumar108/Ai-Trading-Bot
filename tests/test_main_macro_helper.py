"""
Spec Part 2: `main.py::_fetch_macro_daily`, the shared macro-fetch helper
reused by cmd_backtest/cmd_walk_forward/cmd_optimize/cmd_research --
previously an inline block only inside cmd_backtest, factored out so
walk-forward/research/optimize can share it without duplicating the fetch
logic (and so they no longer silently run with macro_context=None).
"""

from __future__ import annotations

from config.settings import Config
from data.market_data import DataUnavailableError, MarketDataProvider
from main import _fetch_macro_daily
from tests.conftest import make_synthetic_ohlcv


def make_config() -> Config:
    return Config()


def test_fetch_macro_daily_disabled_returns_none():
    config = make_config()
    config.macro.enabled = False

    def fetch(symbol, period, interval):
        raise AssertionError("fetch_fn must not be called when macro.enabled=False")

    md = MarketDataProvider(fetch_fn=fetch)
    assert _fetch_macro_daily(config, md, "5y") is None


def test_fetch_macro_daily_provider_none_returns_none():
    config = make_config()
    config.providers.macro = "none"

    def fetch(symbol, period, interval):
        raise AssertionError("fetch_fn must not be called when providers.macro='none'")

    md = MarketDataProvider(fetch_fn=fetch)
    assert _fetch_macro_daily(config, md, "5y") is None


def test_fetch_macro_daily_fetches_all_four_series_when_enabled():
    config = make_config()
    series = make_synthetic_ohlcv(n=100, seed=901)

    def fetch(symbol, period, interval):
        return series.copy()

    md = MarketDataProvider(fetch_fn=fetch)
    result = _fetch_macro_daily(config, md, "5y")
    assert result is not None
    assert set(result.keys()) == {"us_vix", "india_vix", "crude", "usdinr"}
    for df in result.values():
        assert len(df) == 100


def test_fetch_macro_daily_omits_failed_series_and_keeps_others():
    config = make_config()
    series = make_synthetic_ohlcv(n=100, seed=902)

    def fetch(symbol, period, interval):
        if symbol == config.macro.india_vix_symbol:
            raise DataUnavailableError("simulated outage")
        return series.copy()

    md = MarketDataProvider(fetch_fn=fetch)
    result = _fetch_macro_daily(config, md, "5y")
    assert result is not None
    assert "india_vix" not in result
    assert set(result.keys()) == {"us_vix", "crude", "usdinr"}


def test_fetch_macro_daily_all_series_failing_returns_empty_dict_not_none():
    """When macro is enabled but every ticker fails, the result is an EMPTY
    dict, not None -- matching cmd_backtest's original exact behavior
    before this helper was factored out (macro_daily={} was reachable
    there too when every fetch raised)."""
    config = make_config()

    def fetch(symbol, period, interval):
        raise DataUnavailableError("simulated total outage")

    md = MarketDataProvider(fetch_fn=fetch)
    result = _fetch_macro_daily(config, md, "5y")
    assert result == {}


def test_fetch_macro_daily_uses_configured_ticker_symbols():
    config = make_config()
    config.macro.us_vix_symbol = "CUSTOM_VIX"
    series = make_synthetic_ohlcv(n=50, seed=903)
    requested_symbols = []

    def fetch(symbol, period, interval):
        requested_symbols.append(symbol)
        return series.copy()

    md = MarketDataProvider(fetch_fn=fetch)
    _fetch_macro_daily(config, md, "5y")
    assert "CUSTOM_VIX" in requested_symbols
