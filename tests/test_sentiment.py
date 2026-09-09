from datetime import datetime, timedelta, timezone

import pytest

from sentiment.social_sentiment import SocialPost, SocialSentimentAnalyzer, SocialSourceAdapter


class FakeSource(SocialSourceAdapter):
    def __init__(self, posts):
        self._posts = posts

    def fetch(self, symbol, query=None, limit=100):
        return self._posts


def fake_sentiment(text: str) -> float:
    if "moon" in text.lower():
        return 0.9
    if "crash" in text.lower():
        return -0.9
    return 0.0


def make_post(text, author, hours_ago, engagement=10):
    return SocialPost(
        symbol="ACME", text=text, author=author, source="reddit",
        created_at=datetime.now(timezone.utc) - timedelta(hours=hours_ago), engagement=engagement,
    )


def test_no_posts_returns_neutral():
    analyzer = SocialSentimentAnalyzer(source=FakeSource([]), sentiment_fn=fake_sentiment)
    agg = analyzer.analyze("ACME")
    assert agg.mention_volume == 0
    assert agg.social_score() == pytest.approx(50.0)


def test_no_posts_marks_data_explicitly_unavailable_not_neutral():
    """Spec section 6: unavailable social data must be flagged explicitly,
    never silently treated as 'checked and found neutral'."""
    analyzer = SocialSentimentAnalyzer(source=FakeSource([]), sentiment_fn=fake_sentiment)
    agg = analyzer.analyze("ACME")
    assert agg.data_available is False


def test_real_data_marks_data_available_true():
    posts = [make_post("this stock is doing fine", f"user{i}", 1) for i in range(5)]
    analyzer = SocialSentimentAnalyzer(source=FakeSource(posts), sentiment_fn=fake_sentiment)
    agg = analyzer.analyze("ACME")
    assert agg.data_available is True


def test_signal_engine_excludes_social_when_data_unavailable_rather_than_scoring_neutral():
    from config.settings import ConfidenceBands, DecisionThresholds, SignalWeights
    from indicators.technical import TechnicalSnapshot
    from strategy.signal_engine import SignalEngine, SignalInputs

    unavailable_social = SocialSentimentAnalyzer(source=FakeSource([]), sentiment_fn=fake_sentiment).analyze("ACME")
    assert unavailable_social.data_available is False

    technical = TechnicalSnapshot(
        symbol="X", last_close=110, sma20=105, sma50=100, sma200=90, rsi14=58,
        macd_hist=1.2, macd_bullish_cross=True, atr14=2.0, atr_pct_of_price=0.018,
        bb_upper=115, bb_lower=95, bb_percent_b=0.6, obv_trend="RISING", trend="UPTREND",
    )
    engine = SignalEngine(SignalWeights(), DecisionThresholds(min_history_bars=50, min_independent_signals=1), ConfidenceBands())
    inputs = SignalInputs(
        symbol="X", technical=technical, market_trend="UPTREND", relative_strength=0.03,
        fundamentals=None, news=None, social=unavailable_social, avg_volume_20d=500_000,
        min_liquidity_avg_volume=100_000, atr_pct_of_price=0.018, max_atr_pct_of_price=0.08,
        min_atr_pct_of_price=0.002, history_bars=100,
    )
    decision = engine.decide(inputs)
    assert "social_sentiment" not in decision.component_scores
    assert "social_sentiment" in decision.unavailable_components


def test_bot_like_duplicate_posts_detected():
    posts = [make_post("this stock is going to the moon!!!", f"user{i}", 1) for i in range(10)]
    analyzer = SocialSentimentAnalyzer(source=FakeSource(posts), sentiment_fn=fake_sentiment)
    agg = analyzer.analyze("ACME")
    assert agg.bot_like_fraction > 0.5


def test_manipulation_flag_requires_volume_bias_and_bot_pattern():
    # Large volume, one-sided sentiment, near-duplicate text -> manipulation suspected.
    posts = [make_post("this stock is going to the moon!!!", f"user{i}", 1) for i in range(150)]
    analyzer = SocialSentimentAnalyzer(
        source=FakeSource(posts), sentiment_fn=fake_sentiment, min_mentions_for_signal=25,
        spam_bot_score_threshold=0.5,
    )
    agg = analyzer.analyze("ACME")
    assert agg.manipulation_suspected is True
    # Manipulation suspicion must collapse the score toward neutral even
    # though raw sentiment is extremely positive.
    assert agg.social_score() < 70.0


def test_diverse_organic_posts_not_flagged_as_manipulation():
    posts = [make_post(f"I think this company is doing okay, report #{i}", f"user{i}", i % 40) for i in range(30)]
    analyzer = SocialSentimentAnalyzer(source=FakeSource(posts), sentiment_fn=fake_sentiment)
    agg = analyzer.analyze("ACME")
    assert agg.manipulation_suspected is False


def test_sentiment_acceleration_reflects_shift_over_time():
    old_posts = [make_post("stock might crash", f"olduser{i}", 40) for i in range(5)]
    new_posts = [make_post("stock going to the moon", f"newuser{i}", 2) for i in range(5)]
    analyzer = SocialSentimentAnalyzer(source=FakeSource(old_posts + new_posts), sentiment_fn=fake_sentiment, lookback_hours=48)
    agg = analyzer.analyze("ACME")
    assert agg.sentiment_acceleration > 0
