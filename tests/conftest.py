"""
Shared test fixtures.

All tests run fully offline: synthetic OHLCV data is generated
deterministically (fixed random seed) instead of hitting yfinance or any
other network service, so the test suite is fast, reproducible, and works
without internet access.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _no_real_corporate_actions_network_calls(monkeypatch):
    """`MarketDataProvider.__init__` defaults `fetch_fn` and
    `corporate_actions_fetch_fn` INDEPENDENTLY of each other -- a test that
    injects its own offline `fetch_fn` (the overwhelming majority of this
    suite) but never also passes `corporate_actions_fetch_fn` silently
    falls through to `_default_fetch_corporate_actions`, which calls real
    yfinance. That violates this file's own "fully offline" contract
    (slow, network-dependent, and was observed producing real HTTP
    warnings in supposedly-offline test runs). No test in this suite
    exercises the REAL default's behavior against real yfinance -- the two
    tests that care about corporate-actions behavior already inject their
    own explicit `corporate_actions_fetch_fn` (tests/test_market_data.py),
    which this autouse patch does not affect, since a caller-supplied
    value always wins over the constructor's `or _default_fetch_corporate_actions`
    fallback. Patched here once, centrally, rather than touching every
    test file's MarketDataProvider(...) call site individually."""
    def _empty_corporate_actions(symbol: str) -> pd.DataFrame:
        return pd.DataFrame(columns=["date", "type", "value"])

    monkeypatch.setattr(
        "data.market_data._default_fetch_corporate_actions", _empty_corporate_actions,
    )


def make_synthetic_ohlcv(
    n: int = 300,
    start_price: float = 100.0,
    drift: float = 0.0004,
    volatility: float = 0.015,
    seed: int = 42,
    start_date: str = "2023-01-02",
) -> pd.DataFrame:
    """Generates a plausible-looking daily OHLCV series via geometric
    Brownian motion, with a deterministic seed so tests are stable."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=drift, scale=volatility, size=n)
    close = start_price * np.cumprod(1 + returns)

    open_ = np.empty(n)
    open_[0] = start_price
    open_[1:] = close[:-1]

    daily_range = np.abs(rng.normal(loc=volatility * 0.6, scale=volatility * 0.3, size=n)) * close
    high = np.maximum(open_, close) + daily_range * 0.5
    low = np.minimum(open_, close) - daily_range * 0.5
    low = np.clip(low, 0.01, None)

    volume = rng.integers(low=200_000, high=2_000_000, size=n).astype(float)

    dates = pd.bdate_range(start=start_date, periods=n)
    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )
    return df


def make_uptrend_ohlcv(n: int = 300, seed: int = 1) -> pd.DataFrame:
    # Strong, low-noise drift so the trend is unambiguous even in the
    # trailing 20/50-bar windows that trend/regime classifiers look at
    # (a GBM path can meander even with a positive long-run drift; tests
    # need the *tail* of the series to reliably show the trend too).
    return make_synthetic_ohlcv(n=n, drift=0.004, volatility=0.008, seed=seed)


def make_downtrend_ohlcv(n: int = 300, seed: int = 2) -> pd.DataFrame:
    return make_synthetic_ohlcv(n=n, drift=-0.004, volatility=0.008, seed=seed)


def make_choppy_ohlcv(n: int = 300, seed: int = 3) -> pd.DataFrame:
    return make_synthetic_ohlcv(n=n, drift=0.0, volatility=0.008, seed=seed)


def make_high_vol_ohlcv(n: int = 300, seed: int = 4) -> pd.DataFrame:
    return make_synthetic_ohlcv(n=n, drift=0.0, volatility=0.05, seed=seed)


@pytest.fixture
def uptrend_daily():
    return make_uptrend_ohlcv()


@pytest.fixture
def downtrend_daily():
    return make_downtrend_ohlcv()


@pytest.fixture
def choppy_daily():
    return make_choppy_ohlcv()


@pytest.fixture
def high_vol_daily():
    return make_high_vol_ohlcv()


@pytest.fixture
def index_daily():
    return make_synthetic_ohlcv(n=300, drift=0.0006, volatility=0.010, seed=99)
