"""
Global + India macro market intelligence (spec Part 2): "the bot must NOT
analyze a stock in isolation." US VIX / India VIX / crude oil / USD-INR
are just OHLCV-like time series -- the SAME kind of data
`data/market_data.py` already fetches for a benchmark index, unlike
fundamentals/news/social (no free point-in-time dataset exists for those).
So this module fetches them the same way and classifies each series'
trend with `MarketDataProvider.market_trend()`'s existing fast/slow-MA
crossover -- reused unchanged, not reinvented. For VIX/India VIX, an
UPTREND reads as "fear/expected volatility rising"; for crude and USD/INR,
UPTREND means the commodity/currency itself is rising (import-cost and
rupee-weakening headwinds for India respectively).

A short-TTL cache matters here specifically: macro context does not vary
per symbol within one scan cycle, so `MacroDataProvider.get_snapshot()`
fetches once and reuses the result for every symbol scanned within
`cache_ttl_seconds` -- without it, a 50-symbol universe scan would refetch
the SAME 4 market-wide series 50 times for no reason.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from data.market_data import DataUnavailableError, MarketDataProvider

logger = logging.getLogger(__name__)

DEFAULT_MACRO_TICKERS: Dict[str, str] = {
    "us_vix": "^VIX",
    "india_vix": "^INDIAVIX",
    "crude": "CL=F",
    "usdinr": "INR=X",
}


@dataclass
class MacroContext:
    """Each trend is `"UPTREND"|"DOWNTREND"|"SIDEWAYS"|"UNKNOWN"` -- a
    series that failed to fetch or had insufficient history reports
    `"UNKNOWN"` for that series alone (never fabricated, never aborts the
    other three)."""
    us_vix_trend: str = "UNKNOWN"
    india_vix_trend: str = "UNKNOWN"
    crude_trend: str = "UNKNOWN"
    usdinr_trend: str = "UNKNOWN"
    as_of: Optional[pd.Timestamp] = None


class MacroDataProvider:
    def __init__(
        self,
        market_data: Optional[MarketDataProvider] = None,
        tickers: Optional[Dict[str, str]] = None,
        cache_ttl_seconds: float = 300.0,
        period: str = "3mo",
    ):
        self.market_data = market_data or MarketDataProvider()
        self.tickers = dict(tickers) if tickers else dict(DEFAULT_MACRO_TICKERS)
        self.cache_ttl_seconds = cache_ttl_seconds
        self.period = period
        self._cache: Optional[MacroContext] = None
        self._cache_time: Optional[float] = None

    def get_snapshot(self, force_refresh: bool = False) -> MacroContext:
        now = time.monotonic()
        if (
            not force_refresh and self._cache is not None and self._cache_time is not None
            and (now - self._cache_time) < self.cache_ttl_seconds
        ):
            return self._cache

        trends: Dict[str, str] = {}
        as_of: Optional[pd.Timestamp] = None
        for key, ticker in self.tickers.items():
            try:
                daily = self.market_data.get_daily(ticker, period=self.period)
                trends[key] = self.market_data.market_trend(daily)
                if len(daily):
                    candidate = daily.index[-1]
                    if as_of is None or candidate > as_of:
                        as_of = candidate
            except DataUnavailableError as exc:
                logger.warning("Macro series %s (%s) unavailable: %s", key, ticker, exc)
                trends[key] = "UNKNOWN"
            except Exception as exc:
                logger.warning("Macro series %s (%s) failed unexpectedly: %s", key, ticker, exc)
                trends[key] = "UNKNOWN"

        snapshot = MacroContext(
            us_vix_trend=trends.get("us_vix", "UNKNOWN"),
            india_vix_trend=trends.get("india_vix", "UNKNOWN"),
            crude_trend=trends.get("crude", "UNKNOWN"),
            usdinr_trend=trends.get("usdinr", "UNKNOWN"),
            as_of=as_of,
        )
        self._cache = snapshot
        self._cache_time = now
        return snapshot


class NoOpMacroDataProvider(MacroDataProvider):
    """providers.macro = 'none': always reports every series unavailable
    -- the same honest no-op pattern config/providers.py already uses for
    news/social/fundamentals 'none'."""

    def __init__(self):
        super().__init__()

    def get_snapshot(self, force_refresh: bool = False) -> MacroContext:
        return MacroContext()
