"""
Data provider factory (spec section 21).

Maps `config.providers.*` (itself resolved from MARKET_DATA_PROVIDER /
NEWS_PROVIDER / FUNDAMENTALS_PROVIDER / SOCIAL_PROVIDER environment
variables -- see config/settings.py:_apply_env_overrides) to concrete
adapter instances. This is the ONLY place that decision is made, so
swapping a vendor later means adding one branch here, not hunting through
every module that constructs an analyzer.

No API keys live here or anywhere in this repo's source: a real paid-vendor
adapter reads its own credentials from the environment inside its own
`__init__`/fetch method (see .env.example) and is wired in below as a new
branch once implemented -- the "none" branch for every domain is always the
honest no-op (never fabricates data), useful for fully offline/deterministic
runs (tests, demos, CI).
"""

from __future__ import annotations

import logging
from typing import Optional

from config.settings import Config
from data.data_quality import DataQualityChecker
from data.macro_data import MacroDataProvider, NoOpMacroDataProvider
from data.market_calendar import get_calendar
from data.market_data import DataUnavailableError, MarketDataProvider
from fundamentals.fundamental_analysis import FundamentalAnalyzer, FundamentalsUnavailableError
from news.news_analysis import NewsAnalyzer, NewsSourceAdapter, RSSNewsSourceAdapter
from sentiment.social_sentiment import RedditSocialSourceAdapter, SocialSentimentAnalyzer, SocialSourceAdapter

logger = logging.getLogger(__name__)


class NoOpNewsSourceAdapter(NewsSourceAdapter):
    """providers.news = 'none': always reports no news, never fabricated
    sentiment. Use this for fully offline/deterministic runs."""

    def fetch(self, symbol, company_name=None):
        return []


class NoOpSocialSourceAdapter(SocialSourceAdapter):
    """providers.social = 'none': always reports no social data available."""

    def fetch(self, symbol, query=None, limit=100):
        return []


def _no_fundamentals_fetch(symbol: str) -> dict:
    raise FundamentalsUnavailableError(f"providers.fundamentals = 'none' -- no fundamentals fetch configured for {symbol}")


def _no_market_data_fetch(symbol: str, period: str, interval: str):
    raise DataUnavailableError(f"providers.market_data = 'none' -- no market data fetch configured for {symbol}")


def _make_quality_checker(config: Config) -> DataQualityChecker:
    calendar = get_calendar(config.system.trading_calendar)
    dq = config.data_quality
    return DataQualityChecker(
        calendar=calendar,
        max_missing_row_fraction=dq.max_missing_row_fraction,
        max_stale_data_days=dq.max_stale_data_days,
        abnormal_daily_return_threshold=dq.abnormal_daily_return_threshold,
        min_volume_for_liquidity_check=dq.min_volume_for_liquidity_check,
        min_quality_score_to_trade=dq.min_quality_score_to_trade,
    )


def make_market_data_provider(config: Config) -> MarketDataProvider:
    choice = (config.providers.market_data or "yfinance").lower()
    quality_checker = _make_quality_checker(config)
    if choice == "none":
        logger.warning("providers.market_data = 'none': MarketDataProvider will report every symbol unavailable.")
        return MarketDataProvider(fetch_fn=_no_market_data_fetch, quality_checker=quality_checker)
    if choice == "yfinance":
        return MarketDataProvider(quality_checker=quality_checker)  # default fetch_fn is already yfinance-backed
    raise ValueError(f"Unknown providers.market_data={choice!r} (expected 'yfinance' or 'none').")


def make_news_analyzer(config: Config) -> NewsAnalyzer:
    choice = (config.providers.news or "google_rss").lower()
    if choice == "none":
        source: NewsSourceAdapter = NoOpNewsSourceAdapter()
    elif choice == "google_rss":
        source = RSSNewsSourceAdapter()
    else:
        raise ValueError(f"Unknown providers.news={choice!r} (expected 'google_rss' or 'none').")
    return NewsAnalyzer(
        source=source,
        min_source_credibility=config.news.min_source_credibility,
        max_headline_age_hours=config.news.max_headline_age_hours,
        contradiction_window_hours=config.news.contradiction_window_hours,
    )


def make_fundamental_analyzer(config: Config) -> FundamentalAnalyzer:
    choice = (config.providers.fundamentals or "yfinance").lower()
    if choice == "none":
        return FundamentalAnalyzer(fetch_fn=_no_fundamentals_fetch)
    if choice == "yfinance":
        return FundamentalAnalyzer()
    raise ValueError(f"Unknown providers.fundamentals={choice!r} (expected 'yfinance' or 'none').")


def make_macro_data_provider(config: Config, market_data: Optional[MarketDataProvider] = None) -> MacroDataProvider:
    """
    `market_data`: reuse an ALREADY-resolved `MarketDataProvider` (real or,
    critically, a test's injected fake `fetch_fn`) rather than constructing
    a brand-new network-backed one internally -- `PaperTradingEngine`
    always passes its own `self.market_data` here, so any caller/test that
    injects a fake market-data fetch function automatically gets safe,
    deterministic, offline macro behavior too (the fake is called with the
    macro tickers as `symbol`, same as any other symbol). Without this, a
    macro-off-by-default design would make EVERY existing offline test
    that constructs a PaperTradingEngine silently start hitting the
    network for VIX/crude/INR data.
    """
    choice = (config.providers.macro or "yfinance").lower()
    if choice == "none" or not config.macro.enabled:
        return NoOpMacroDataProvider()
    if choice == "yfinance":
        tickers = {
            "us_vix": config.macro.us_vix_symbol, "india_vix": config.macro.india_vix_symbol,
            "crude": config.macro.crude_symbol, "usdinr": config.macro.usdinr_symbol,
        }
        return MacroDataProvider(
            market_data=market_data, tickers=tickers, cache_ttl_seconds=config.macro.cache_ttl_seconds,
        )
    raise ValueError(f"Unknown providers.macro={choice!r} (expected 'yfinance' or 'none').")


def make_social_analyzer(config: Config) -> SocialSentimentAnalyzer:
    choice = (config.providers.social or "reddit").lower()
    if choice == "none":
        source: SocialSourceAdapter = NoOpSocialSourceAdapter()
    elif choice == "reddit":
        source = RedditSocialSourceAdapter()
    else:
        raise ValueError(f"Unknown providers.social={choice!r} (expected 'reddit' or 'none').")
    return SocialSentimentAnalyzer(
        source=source,
        min_mentions_for_signal=config.social.min_mentions_for_signal,
        spam_bot_score_threshold=config.social.spam_bot_score_threshold,
    )
