"""
Spec Part 2: global + India macro market intelligence.

Covers `MacroDataProvider` fetch/cache/failure-isolation behavior with an
injected fake `MarketDataProvider` (no real network calls), and
`config/providers.py::make_macro_data_provider`'s "none"/disabled no-op
path.
"""

from __future__ import annotations

import time

import pandas as pd
import pytest

from data.macro_data import DEFAULT_MACRO_TICKERS, MacroContext, MacroDataProvider, NoOpMacroDataProvider
from data.market_data import DataUnavailableError, MarketDataProvider
from tests.conftest import make_downtrend_ohlcv, make_uptrend_ohlcv


def _fake_market_data(series_by_symbol):
    def fetch(symbol, period, interval):
        if symbol not in series_by_symbol:
            raise DataUnavailableError(f"no fake data for {symbol}")
        return series_by_symbol[symbol].copy()
    return MarketDataProvider(fetch_fn=fetch)


def test_snapshot_classifies_each_series_trend():
    up = make_uptrend_ohlcv(n=300, seed=401)
    down = make_downtrend_ohlcv(n=300, seed=402)
    md = _fake_market_data({
        "^VIX": down, "^INDIAVIX": down, "CL=F": up, "INR=X": up,
    })
    provider = MacroDataProvider(market_data=md)
    snap = provider.get_snapshot()
    assert snap.us_vix_trend == "DOWNTREND"
    assert snap.india_vix_trend == "DOWNTREND"
    assert snap.crude_trend == "UPTREND"
    assert snap.usdinr_trend == "UPTREND"
    assert snap.as_of is not None


def test_one_series_failure_does_not_break_the_others():
    up = make_uptrend_ohlcv(n=300, seed=403)
    md = _fake_market_data({"^VIX": up, "CL=F": up, "INR=X": up})  # ^INDIAVIX missing -> DataUnavailableError
    provider = MacroDataProvider(market_data=md)
    snap = provider.get_snapshot()
    assert snap.india_vix_trend == "UNKNOWN"
    assert snap.us_vix_trend == "UPTREND"  # the other three still resolve


def test_unexpected_exception_in_one_series_is_isolated():
    up = make_uptrend_ohlcv(n=300, seed=404)

    def fetch(symbol, period, interval):
        if symbol == "CL=F":
            raise RuntimeError("simulated vendor crash")
        return up.copy()

    provider = MacroDataProvider(market_data=MarketDataProvider(fetch_fn=fetch))
    snap = provider.get_snapshot()
    assert snap.crude_trend == "UNKNOWN"
    assert snap.us_vix_trend == "UPTREND"


def test_snapshot_is_cached_within_ttl():
    up = make_uptrend_ohlcv(n=300, seed=405)
    call_count = {"n": 0}

    def fetch(symbol, period, interval):
        call_count["n"] += 1
        return up.copy()

    provider = MacroDataProvider(market_data=MarketDataProvider(fetch_fn=fetch), cache_ttl_seconds=1000.0)
    provider.get_snapshot()
    calls_after_first = call_count["n"]
    provider.get_snapshot()  # should be served from cache -- no new fetches
    assert call_count["n"] == calls_after_first


def test_force_refresh_recomputes_the_snapshot():
    """`force_refresh` bypasses MacroDataProvider's OWN snapshot cache --
    the underlying MarketDataProvider may still serve its own per-symbol
    cache (a separate, lower layer; see MarketDataProvider.get_daily),
    so this only asserts the snapshot-level cache timestamp actually
    advances, not that a real re-fetch necessarily happens."""
    up = make_uptrend_ohlcv(n=300, seed=406)
    md = MarketDataProvider(fetch_fn=lambda symbol, period, interval: up.copy())
    provider = MacroDataProvider(market_data=md, cache_ttl_seconds=1000.0)
    provider.get_snapshot()
    first_cache_time = provider._cache_time
    time.sleep(0.01)
    provider.get_snapshot(force_refresh=True)
    assert provider._cache_time > first_cache_time


def test_custom_tickers_are_used():
    up = make_uptrend_ohlcv(n=300, seed=407)
    md = _fake_market_data({"CUSTOM_VIX": up, "^INDIAVIX": up, "CL=F": up, "INR=X": up})
    provider = MacroDataProvider(market_data=md, tickers={**DEFAULT_MACRO_TICKERS, "us_vix": "CUSTOM_VIX"})
    snap = provider.get_snapshot()
    assert snap.us_vix_trend == "UPTREND"


def test_noop_provider_reports_everything_unknown():
    provider = NoOpMacroDataProvider()
    snap = provider.get_snapshot()
    assert snap == MacroContext()  # all UNKNOWN, never fabricated


# =============================================================================
# config/providers.py wiring
# =============================================================================

def test_make_macro_data_provider_none_choice_is_noop():
    from config.providers import make_macro_data_provider
    from config.settings import Config

    config = Config()
    config.providers.macro = "none"
    provider = make_macro_data_provider(config)
    assert isinstance(provider, NoOpMacroDataProvider)


def test_make_macro_data_provider_disabled_is_noop():
    from config.providers import make_macro_data_provider
    from config.settings import Config

    config = Config()
    config.macro.enabled = False
    provider = make_macro_data_provider(config)
    assert isinstance(provider, NoOpMacroDataProvider)


def test_make_macro_data_provider_reuses_injected_market_data():
    """Critical: must reuse the CALLER's market_data (real or fake), never
    silently construct its own network-backed MarketDataProvider --
    otherwise every offline test/caller that injects a fake fetch_fn would
    unexpectedly hit the real network for macro tickers."""
    from config.providers import make_macro_data_provider
    from config.settings import Config

    up = make_uptrend_ohlcv(n=300, seed=408)
    fake_md = _fake_market_data({
        "^VIX": up, "^INDIAVIX": up, "CL=F": up, "INR=X": up,
    })
    config = Config()
    provider = make_macro_data_provider(config, market_data=fake_md)
    assert provider.market_data is fake_md
    snap = provider.get_snapshot()
    assert snap.us_vix_trend == "UPTREND"  # resolved via the FAKE, not a real network call


def test_make_macro_data_provider_unknown_choice_raises():
    from config.providers import make_macro_data_provider
    from config.settings import Config

    config = Config()
    config.providers.macro = "bogus_vendor"
    with pytest.raises(ValueError):
        make_macro_data_provider(config)
