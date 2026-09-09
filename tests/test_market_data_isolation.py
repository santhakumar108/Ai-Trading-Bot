"""
Spec section 8: 'Add a test proving that two different tickers do not
accidentally share the same market-data series.'

MarketDataProvider caches fetched frames keyed by f"{symbol}:daily:{period}"
(data/market_data.py). This test proves that keying is actually symbol-
specific -- two different tickers fetched through the SAME provider instance
get distinct DataFrame objects with distinct values, and mutating one
in-place (via the caller's own copy) never affects the other or the
provider's internal cache.
"""

from __future__ import annotations

import pandas as pd
import pytest

from data.market_data import MarketDataProvider
from tests.conftest import make_synthetic_ohlcv


def make_per_symbol_fetch_fn():
    """A fetch_fn that returns a DIFFERENT, symbol-specific series per
    ticker -- if the provider ever accidentally reused one series for every
    symbol (e.g. a caching bug keyed only by period, or a shared mutable
    default), this would catch it immediately."""
    series_by_symbol = {
        "TCS.NS": make_synthetic_ohlcv(n=100, drift=0.003, volatility=0.01, seed=1, start_date="2022-01-03"),
        "INFY.NS": make_synthetic_ohlcv(n=100, drift=-0.002, volatility=0.02, seed=2, start_date="2022-01-03"),
    }

    def fetch(symbol, period, interval):
        return series_by_symbol[symbol].copy()

    return fetch, series_by_symbol


def test_two_tickers_get_distinct_dataframes_from_the_same_provider():
    fetch_fn, series_by_symbol = make_per_symbol_fetch_fn()
    provider = MarketDataProvider(fetch_fn=fetch_fn)

    tcs = provider.get_daily("TCS.NS")
    infy = provider.get_daily("INFY.NS")

    assert tcs is not infy
    assert not tcs["Close"].equals(infy["Close"])
    pd.testing.assert_series_equal(tcs["Close"].reset_index(drop=True), series_by_symbol["TCS.NS"]["Close"].reset_index(drop=True))
    pd.testing.assert_series_equal(infy["Close"].reset_index(drop=True), series_by_symbol["INFY.NS"]["Close"].reset_index(drop=True))


def test_cache_keys_are_symbol_specific_not_shared():
    fetch_fn, _ = make_per_symbol_fetch_fn()
    provider = MarketDataProvider(fetch_fn=fetch_fn)

    provider.get_daily("TCS.NS")
    provider.get_daily("INFY.NS")

    assert "TCS.NS:daily:2y" in provider._cache
    assert "INFY.NS:daily:2y" in provider._cache
    assert not provider._cache["TCS.NS:daily:2y"]["Close"].equals(provider._cache["INFY.NS:daily:2y"]["Close"])


def test_mutating_one_symbols_returned_frame_never_affects_the_other():
    """`get_daily` returns `.copy()` of the cached frame (data/market_data.py)
    -- the caller mutating its own copy must never leak into the provider's
    cache for a DIFFERENT symbol (or the same symbol, on a subsequent call)."""
    fetch_fn, _ = make_per_symbol_fetch_fn()
    provider = MarketDataProvider(fetch_fn=fetch_fn)

    tcs_first = provider.get_daily("TCS.NS")
    tcs_first["Close"] = 999999.0  # mutate the caller's own copy

    tcs_second = provider.get_daily("TCS.NS")
    infy = provider.get_daily("INFY.NS")

    assert not (tcs_second["Close"] == 999999.0).all()
    assert not (infy["Close"] == 999999.0).any()


def test_repeated_fetch_of_same_symbol_is_cached_and_stable():
    """Not strictly section 8, but the flip side of the same guarantee: the
    SAME symbol fetched twice must return equal data (from cache, not a
    fresh possibly-different fetch), so a caller can't observe a ticker's
    own history silently changing mid-run."""
    fetch_fn, _ = make_per_symbol_fetch_fn()
    provider = MarketDataProvider(fetch_fn=fetch_fn)

    first = provider.get_daily("TCS.NS")
    second = provider.get_daily("TCS.NS")
    assert first is not second  # distinct copies...
    pd.testing.assert_frame_equal(first, second)  # ...but equal content


def test_two_tickers_produce_independent_technical_snapshots():
    """End-to-end version of the isolation guarantee: running the real
    TechnicalAnalyzer against each symbol's own fetched series must never
    show cross-contaminated values (e.g. TCS's indicators computed off
    INFY's prices due to a mix-up)."""
    from indicators.technical import TechnicalAnalyzer

    fetch_fn, _ = make_per_symbol_fetch_fn()
    provider = MarketDataProvider(fetch_fn=fetch_fn)
    analyzer = TechnicalAnalyzer()

    tcs_daily = provider.get_daily("TCS.NS")
    infy_daily = provider.get_daily("INFY.NS")

    tcs_snap = analyzer.analyze("TCS.NS", tcs_daily)
    infy_snap = analyzer.analyze("INFY.NS", infy_daily)

    assert tcs_snap.last_close == pytest.approx(float(tcs_daily["Close"].iloc[-1]))
    assert infy_snap.last_close == pytest.approx(float(infy_daily["Close"].iloc[-1]))
    assert tcs_snap.last_close != pytest.approx(infy_snap.last_close)
