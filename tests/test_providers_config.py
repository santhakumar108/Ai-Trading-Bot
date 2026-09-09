import pytest

from config.providers import (
    make_fundamental_analyzer, make_market_data_provider, make_news_analyzer, make_social_analyzer,
)
from config.settings import Config
from data.market_data import DataUnavailableError
from fundamentals.fundamental_analysis import FundamentalsUnavailableError


def make_config(**provider_overrides) -> Config:
    config = Config()
    for k, v in provider_overrides.items():
        setattr(config.providers, k, v)
    return config


def test_market_data_none_provider_always_reports_unavailable():
    config = make_config(market_data="none")
    provider = make_market_data_provider(config)
    with pytest.raises(DataUnavailableError):
        provider.get_daily("RELIANCE.NS")


def test_market_data_yfinance_is_the_default():
    config = make_config()
    provider = make_market_data_provider(config)
    assert provider.fetch_fn.__name__ == "_default_fetch_history"


def test_news_none_provider_returns_no_articles_not_fabricated():
    config = make_config(news="none")
    analyzer = make_news_analyzer(config)
    agg = analyzer.analyze("RELIANCE.NS")
    assert agg.items == []
    assert agg.overall_confidence == 0.0


def test_fundamentals_none_provider_leaves_everything_missing():
    config = make_config(fundamentals="none")
    analyzer = make_fundamental_analyzer(config)
    snapshot = analyzer.analyze("RELIANCE.NS")
    assert snapshot.data_quality == 0.0
    assert snapshot.revenue_growth is None
    assert snapshot.pe_ratio is None


def test_social_none_provider_marks_data_unavailable():
    config = make_config(social="none")
    analyzer = make_social_analyzer(config)
    agg = analyzer.analyze("RELIANCE.NS")
    assert agg.data_available is False
    assert agg.mention_volume == 0


def test_unknown_provider_choice_raises_rather_than_silently_falling_back():
    config = make_config(market_data="some_paid_vendor_not_implemented")
    with pytest.raises(ValueError):
        make_market_data_provider(config)
