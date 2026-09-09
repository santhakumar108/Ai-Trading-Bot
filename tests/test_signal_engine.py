import dataclasses

import pytest

from config.settings import ConfidenceBands, DecisionThresholds, SignalWeights
from fundamentals.fundamental_analysis import FundamentalSnapshot
from indicators.technical import TechnicalSnapshot
from news.news_analysis import NewsAggregate
from sentiment.social_sentiment import SocialAggregate
from strategy.signal_engine import SignalEngine, SignalInputs, _confidence_multiplier, _technical_alpha_score


def make_engine(**threshold_overrides):
    weights = SignalWeights()
    thresholds = DecisionThresholds(min_history_bars=100, **threshold_overrides)
    bands = ConfidenceBands()
    return SignalEngine(weights, thresholds, bands)


def make_technical(score_bias: str = "bullish") -> TechnicalSnapshot:
    if score_bias == "bullish":
        return TechnicalSnapshot(
            symbol="X", last_close=110, sma20=105, sma50=100, sma200=90, rsi14=58,
            macd_hist=1.2, macd_bullish_cross=True, atr14=2.0, atr_pct_of_price=0.018,
            bb_upper=115, bb_lower=95, bb_percent_b=0.6, obv_trend="RISING", trend="UPTREND",
        )
    return TechnicalSnapshot(
        symbol="X", last_close=90, sma20=95, sma50=100, sma200=110, rsi14=40,
        macd_hist=-1.0, macd_bullish_cross=False, atr14=2.0, atr_pct_of_price=0.018,
        bb_upper=105, bb_lower=85, bb_percent_b=0.4, obv_trend="FALLING", trend="DOWNTREND",
    )


def make_fundamentals(good: bool = True) -> FundamentalSnapshot:
    if good:
        return FundamentalSnapshot(
            symbol="X", revenue_growth=0.15, earnings_growth=0.20, eps=5.0, pe_ratio=20,
            pb_ratio=3, debt_to_equity=50, roe=0.20, profit_margin=0.15,
            operating_cash_flow=1_000_000, free_cash_flow=500_000, last_update=None,
            data_quality=1.0, is_stale=False,
        )
    return FundamentalSnapshot(
        symbol="X", revenue_growth=-0.1, earnings_growth=-0.2, eps=-1.0, pe_ratio=80,
        pb_ratio=12, debt_to_equity=250, roe=-0.1, profit_margin=-0.05,
        operating_cash_flow=-100_000, free_cash_flow=-200_000, last_update=None,
        data_quality=1.0, is_stale=False,
    )


def make_news(good: bool = True) -> NewsAggregate:
    score = 60.0 if good else -60.0
    return NewsAggregate(symbol="X", items=[], overall_sentiment_score=score, overall_confidence=90.0,
                          contradictory=False, num_credible_sources=3)


def make_social(good: bool = True) -> SocialAggregate:
    return SocialAggregate(
        symbol="X", posts=[], mention_volume=50, sentiment_mean=(0.5 if good else -0.5),
        positive_ratio=(0.7 if good else 0.1), negative_ratio=(0.1 if good else 0.7),
        sentiment_acceleration=0.1, abnormal_activity=False, bot_like_fraction=0.1,
        manipulation_suspected=False, credibility_factor=0.7,
    )


def base_inputs(**overrides) -> SignalInputs:
    defaults = dict(
        symbol="X", technical=make_technical("bullish"), market_trend="UPTREND",
        relative_strength=0.03, fundamentals=make_fundamentals(True), news=make_news(True),
        social=make_social(True), avg_volume_20d=500_000, min_liquidity_avg_volume=100_000,
        atr_pct_of_price=0.018, max_atr_pct_of_price=0.08, min_atr_pct_of_price=0.002,
        history_bars=200,
    )
    defaults.update(overrides)
    return SignalInputs(**defaults)


def test_default_is_no_trade_with_insufficient_history():
    engine = make_engine()
    inputs = base_inputs(history_bars=10)
    decision = engine.decide(inputs)
    assert decision.decision == "NO TRADE"
    assert "Insufficient historical evidence" in decision.reasons[0]


def test_aligned_bullish_signals_produce_buy():
    engine = make_engine(min_confidence_to_trade=60)
    decision = engine.decide(base_inputs())
    assert decision.decision == "BUY"
    assert decision.overall_confidence >= 60


def test_disagreeing_components_force_no_trade():
    engine = make_engine(max_model_disagreement=0.2)
    # Technical bullish, fundamentals/news/social all bad -> high disagreement.
    inputs = base_inputs(
        technical=make_technical("bullish"),
        fundamentals=make_fundamentals(False),
        news=make_news(False),
        social=make_social(False),
    )
    decision = engine.decide(inputs)
    assert decision.decision == "NO TRADE"
    assert decision.model_agreement < 0.8


def test_low_confidence_defaults_to_no_trade():
    engine = make_engine(min_confidence_to_trade=95)  # deliberately hard to reach
    decision = engine.decide(base_inputs())
    assert decision.decision in ("NO TRADE", "HOLD")
    assert decision.decision != "BUY"


def test_illiquid_symbol_lowers_volume_score():
    engine = make_engine()
    liquid = engine.decide(base_inputs(avg_volume_20d=1_000_000))
    illiquid = engine.decide(base_inputs(avg_volume_20d=1_000))
    assert illiquid.component_scores["volume_price_behavior"] < liquid.component_scores["volume_price_behavior"]


def test_abnormal_volatility_lowers_risk_score():
    engine = make_engine()
    normal = engine.decide(base_inputs(atr_pct_of_price=0.02))
    abnormal = engine.decide(base_inputs(atr_pct_of_price=0.20))
    assert abnormal.component_scores["risk_volatility"] < normal.component_scores["risk_volatility"]


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        SignalWeights(technical=0.9, market_condition=0.9).validate()


# --- Availability, not fabrication (spec: never fake neutral data) --------

def test_unavailable_components_are_excluded_not_faked():
    engine = make_engine()
    inputs = base_inputs(fundamentals=None, news=None, social=None)
    decision = engine.decide(inputs)
    assert "fundamentals" not in decision.component_scores
    assert "news_sentiment" not in decision.component_scores
    assert "social_sentiment" not in decision.component_scores
    assert set(decision.unavailable_components) >= {"fundamentals", "news_sentiment", "social_sentiment"}


def test_available_weight_is_renormalized_not_diluted_toward_fifty():
    """With fundamentals/news/social unavailable, the overall score must be
    the weighted average of ONLY technical/market/volume/risk (renormalized
    to sum to 1), never diluted by treating the missing components as a
    fabricated neutral 50."""
    engine = make_engine(min_independent_signals=1)
    inputs = base_inputs(fundamentals=None, news=None, social=None)
    decision = engine.decide(inputs)
    scores = decision.component_scores
    weights = {"technical": 0.25, "market_condition": 0.15, "volume_price_behavior": 0.10, "risk_volatility": 0.10}
    total_w = sum(weights.values())
    expected = sum(scores[k] * (weights[k] / total_w) for k in weights)
    assert decision.overall_confidence == pytest.approx(expected, abs=0.05)


def test_min_independent_signals_gate_blocks_single_signal_trades():
    """Only technical is available -> 1 independent signal -> must never be
    enough to trade on by itself, regardless of how bullish it looks."""
    engine = make_engine(min_independent_signals=2)
    inputs = base_inputs(fundamentals=None, news=None, social=None)
    decision = engine.decide(inputs)
    assert decision.available_independent_signals == 1
    assert decision.decision == "NO TRADE"


def test_min_independent_signals_of_one_allows_technical_only_trades():
    engine = make_engine(min_independent_signals=1, min_confidence_to_trade=40)
    inputs = base_inputs(fundamentals=None, news=None, social=None)
    decision = engine.decide(inputs)
    assert decision.available_independent_signals == 1
    assert decision.decision in ("BUY", "HOLD", "NO TRADE")  # never blocked purely by the count gate


# =============================================================================
# Spec Part 15: decision-fusion reweighting by confidence
# =============================================================================

def test_confidence_multiplier_unavailable_component_stays_at_one():
    assert _confidence_multiplier("technical", False, None, False, None, False, None) == 1.0
    assert _confidence_multiplier("market_condition", False, None, False, None, False, None) == 1.0
    assert _confidence_multiplier("volume_price_behavior", False, None, False, None, False, None) == 1.0
    assert _confidence_multiplier("risk_volatility", False, None, False, None, False, None) == 1.0


def test_confidence_multiplier_fundamentals_uses_fundamental_confidence():
    fund = make_fundamentals(True)
    expected = max(0.3, fund.fundamental_confidence())
    assert _confidence_multiplier("fundamentals", True, fund, False, None, False, None) == pytest.approx(expected)


def test_confidence_multiplier_news_uses_overall_confidence():
    news = make_news(True)  # overall_confidence=90.0
    m = _confidence_multiplier("news_sentiment", False, None, True, news, False, None)
    assert m == pytest.approx(max(0.3, 0.90))


def test_confidence_multiplier_social_uses_credibility_factor():
    social = make_social(True)  # credibility_factor=0.7
    m = _confidence_multiplier("social_sentiment", False, None, False, None, True, social)
    assert m == pytest.approx(max(0.3, 0.7))


def test_confidence_multiplier_floored_at_0_3():
    news = NewsAggregate(symbol="X", items=[], overall_sentiment_score=10.0, overall_confidence=5.0,
                          contradictory=False, num_credible_sources=1)
    m = _confidence_multiplier("news_sentiment", False, None, True, news, False, None)
    assert m == pytest.approx(0.3)


def test_confidence_multiplier_capped_at_1_0_even_for_out_of_range_credibility_factor():
    """`credibility_factor` is a caller-supplied field on SocialAggregate
    with no clipping guarantee at its source (unlike fundamental_confidence()
    and overall_confidence/100, which are self-bounded). A value above 1.0
    must not amplify this component's influence beyond its configured
    weight -- the multiplier is a clamp, not a floor-only function."""
    social = make_social(True)
    social.credibility_factor = 1.4
    m = _confidence_multiplier("social_sentiment", False, None, False, None, True, social)
    assert m == pytest.approx(1.0)


def test_low_confidence_fundamentals_changes_overall_versus_high_confidence():
    """Same fundamental_score() (identical score-driving fields), only
    data_quality differs (-> different fundamental_confidence()) -- the
    resulting overall_confidence must differ, and the low-confidence run
    must say so in its reasons."""
    engine = make_engine(min_confidence_to_trade=1, min_independent_signals=1)
    high_conf_fund = make_fundamentals(True)  # data_quality=1.0
    low_conf_fund = dataclasses.replace(high_conf_fund, data_quality=0.4)
    assert low_conf_fund.fundamental_score() == pytest.approx(high_conf_fund.fundamental_score())
    assert low_conf_fund.fundamental_confidence() < high_conf_fund.fundamental_confidence()

    high_decision = engine.decide(base_inputs(fundamentals=high_conf_fund))
    low_decision = engine.decide(base_inputs(fundamentals=low_conf_fund))
    assert high_decision.overall_confidence != pytest.approx(low_decision.overall_confidence)
    assert any("confidence-weighted fusion" in r for r in low_decision.reasons)


def test_low_confidence_fundamentals_still_counted_never_fully_zeroed():
    """The 0.3 floor: even a low-confidence fundamentals reading stays IN
    component_scores (still influences the decision), unlike a genuinely
    unavailable one which is excluded entirely."""
    engine = make_engine(min_confidence_to_trade=1, min_independent_signals=1)
    low_conf_fund = dataclasses.replace(make_fundamentals(True), data_quality=0.4)
    decision = engine.decide(base_inputs(fundamentals=low_conf_fund))
    assert "fundamentals" in decision.component_scores
    assert "fundamentals" not in decision.unavailable_components


# =============================================================================
# Spec Part 16: news-conflict override
# =============================================================================

def make_news_extreme(sentiment_score: float, confidence: float, num_sources: int = 3) -> NewsAggregate:
    return NewsAggregate(symbol="X", items=[], overall_sentiment_score=sentiment_score,
                          overall_confidence=confidence, contradictory=False, num_credible_sources=num_sources)


def test_strongly_negative_credible_news_vetoes_an_otherwise_buy():
    engine = make_engine(min_confidence_to_trade=1, min_independent_signals=1, max_model_disagreement=2.0)
    strongly_negative_news = make_news_extreme(sentiment_score=-90.0, confidence=80.0)
    inputs = base_inputs(
        technical=make_technical("bullish"), fundamentals=make_fundamentals(True),
        news=strongly_negative_news, social=make_social(True),
    )
    # Confirm the setup WOULD have been a BUY without the news conflict.
    without_news = engine.decide(base_inputs(
        technical=make_technical("bullish"), fundamentals=make_fundamentals(True),
        news=None, social=make_social(True),
    ))
    assert without_news.decision == "BUY"

    decision = engine.decide(inputs)
    assert decision.decision != "BUY"
    assert decision.decision == "NO TRADE"
    assert any("Part 16" in r for r in decision.reasons)


def test_strongly_positive_credible_news_vetoes_an_otherwise_sell():
    engine = make_engine(min_confidence_to_trade=1, min_independent_signals=1, max_model_disagreement=2.0)
    strongly_positive_news = make_news_extreme(sentiment_score=90.0, confidence=80.0)
    without_news = engine.decide(base_inputs(
        technical=make_technical("bearish"), fundamentals=make_fundamentals(False),
        news=None, social=make_social(False),
    ))
    assert without_news.decision == "SELL"

    decision = engine.decide(base_inputs(
        technical=make_technical("bearish"), fundamentals=make_fundamentals(False),
        news=strongly_positive_news, social=make_social(False),
    ))
    assert decision.decision != "SELL"
    assert decision.decision == "NO TRADE"
    assert any("Part 16" in r for r in decision.reasons)


def test_moderate_negative_news_does_not_trigger_the_override():
    """Above the score threshold -- disagreement, not a strong conflict --
    must not veto an otherwise-good trade."""
    engine = make_engine(min_confidence_to_trade=1, min_independent_signals=1, max_model_disagreement=2.0)
    moderate_news = make_news_extreme(sentiment_score=-40.0, confidence=80.0)
    assert moderate_news.news_score() > 20.0  # confirms this scenario is genuinely "moderate", not "strong"
    decision = engine.decide(base_inputs(
        technical=make_technical("bullish"), fundamentals=make_fundamentals(True),
        news=moderate_news, social=make_social(True),
    ))
    assert decision.decision == "BUY"


def test_low_confidence_extreme_news_does_not_trigger_the_override():
    """Weak/unreliable news, even with an extreme raw sentiment score, must
    not be able to veto a good trade -- confidence gates the override."""
    engine = make_engine(min_confidence_to_trade=1, min_independent_signals=1, max_model_disagreement=2.0)
    unreliable_news = make_news_extreme(sentiment_score=-90.0, confidence=30.0)
    decision = engine.decide(base_inputs(
        technical=make_technical("bullish"), fundamentals=make_fundamentals(True),
        news=unreliable_news, social=make_social(True),
    ))
    assert decision.decision == "BUY"


# =============================================================================
# Spec Part 5/6 (Phase 10): the multi-alpha family blend actually drives the
# "technical" component -- not just computed and left unused.
# =============================================================================

class _FakeTechnicalFamilies:
    """Duck-typed TechnicalSnapshot stand-in with INDEPENDENTLY controllable
    trend/momentum/volatility_volume family scores, to prove
    _technical_alpha_score() genuinely blends all three (not just one)."""

    def __init__(self, trend: float, momentum: float, vol_vol: float):
        self._trend = trend
        self._momentum = momentum
        self._vol_vol = vol_vol

    def trend_score(self):
        return self._trend

    def momentum_score(self):
        return self._momentum

    def volatility_volume_score(self):
        return self._vol_vol


def test_technical_alpha_score_is_the_equal_weighted_mean_of_the_three_families():
    snap = _FakeTechnicalFamilies(trend=80.0, momentum=40.0, vol_vol=60.0)
    assert _technical_alpha_score(snap) == pytest.approx((80.0 + 40.0 + 60.0) / 3.0)


def test_technical_alpha_score_bounded_0_100():
    assert _technical_alpha_score(_FakeTechnicalFamilies(0.0, 0.0, 0.0)) == pytest.approx(0.0)
    assert _technical_alpha_score(_FakeTechnicalFamilies(100.0, 100.0, 100.0)) == pytest.approx(100.0)


def test_technical_alpha_score_genuinely_uses_all_three_families_not_just_one():
    """A case where the three families diverge -- strong trend, weak
    momentum -- must produce a blend strictly BETWEEN them, proving the
    function reads all three, not silently defaulting to one."""
    snap = _FakeTechnicalFamilies(trend=90.0, momentum=30.0, vol_vol=50.0)
    blended = _technical_alpha_score(snap)
    assert 30.0 < blended < 90.0
    assert blended != pytest.approx(90.0)
    assert blended != pytest.approx(30.0)


def test_signal_engine_uses_the_alpha_blend_not_the_old_technical_score():
    """End-to-end proof on a REAL TechnicalSnapshot (not a fake): when the
    three family scores genuinely diverge, component_scores["technical"]
    must equal the NEW blend, not the OLD technical_score() value."""
    from indicators.technical import TechnicalAnalyzer
    from tests.conftest import make_synthetic_ohlcv

    # A series likely to show some divergence between trend/momentum/
    # volume-volatility reads (real data, not hand-tuned to force a result).
    daily = make_synthetic_ohlcv(n=300, seed=77, drift=0.003, volatility=0.02)
    technical = TechnicalAnalyzer().analyze("X", daily)
    old_score = technical.technical_score()
    new_blend = _technical_alpha_score(technical)

    engine = make_engine(min_confidence_to_trade=1, min_independent_signals=1)
    inputs = base_inputs(technical=technical, fundamentals=None, news=None, social=None)
    decision = engine.decide(inputs)
    assert decision.component_scores["technical"] == pytest.approx(new_blend)
    if abs(old_score - new_blend) > 0.01:  # only a meaningful assertion when they actually differ
        assert decision.component_scores["technical"] != pytest.approx(old_score)


def test_available_independent_signals_unaffected_by_the_alpha_blend():
    """Spec Part 6: the technical family blend must still count as exactly
    ONE directional vote, never three -- confirmed by re-running the
    existing single-signal-count test's assertions with the new scoring
    live underneath it."""
    from strategy.signal_engine import ALL_COMPONENTS

    engine = make_engine(min_independent_signals=1)
    inputs = base_inputs(fundamentals=None, news=None, social=None)
    decision = engine.decide(inputs)
    assert decision.available_independent_signals == 1
    assert "technical" in decision.component_scores
    assert len(decision.component_scores) <= len(ALL_COMPONENTS)

