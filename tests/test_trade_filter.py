import pytest

from risk.risk_engine import TradeRiskCalculator
from strategy.signal_engine import SignalDecision
from strategy.trade_filter import TradeFilter


def make_filter(**overrides):
    defaults = dict(
        min_risk_reward=2.0, min_liquidity_avg_volume=100_000, max_atr_pct_of_price=0.08,
        min_atr_pct_of_price=0.002, min_confidence_to_trade=70, max_model_disagreement=0.35,
    )
    defaults.update(overrides)
    return TradeFilter(**defaults)


def make_buy_signal(confidence=80.0, agreement=0.9) -> SignalDecision:
    return SignalDecision(
        symbol="X",
        component_scores={"technical": 75, "market_condition": 65, "fundamentals": 70, "news_sentiment": 68,
                           "social_sentiment": 60, "volume_price_behavior": 70, "risk_volatility": 65},
        overall_confidence=confidence, confidence_label="HIGH CONFIDENCE", direction="UP",
        model_agreement=agreement, decision="BUY", reasons=["ok"],
    )


def make_risk(rr=2.5, size=10):
    calc = TradeRiskCalculator(
        risk_per_trade_pct=0.01, min_risk_reward=2.0, transaction_cost_pct=0.001,
        slippage_pct=0.0007, min_edge_after_costs_pct=0.001, max_single_position_pct=0.2,
    )
    entry = 100
    stop = 97
    target = entry + (entry - stop) * rr
    return calc.evaluate("X", "BUY", entry=entry, stop_loss=stop, target=target, capital=1_000_000)


def test_all_checks_pass_approves_trade():
    tf = make_filter()
    result = tf.approve(
        signal=make_buy_signal(), risk=make_risk(rr=2.67), avg_volume_20d=500_000, atr_pct_of_price=0.02,
        account_level_reasons=[], news_credible_sources=2, news_contradictory=False,
        social_manipulation_suspected=False,
    )
    assert result.approved is True
    assert result.final_decision == "BUY"


def test_low_liquidity_forces_no_trade():
    tf = make_filter()
    result = tf.approve(
        signal=make_buy_signal(), risk=make_risk(rr=2.67), avg_volume_20d=1_000, atr_pct_of_price=0.02,
        account_level_reasons=[], news_credible_sources=2, news_contradictory=False,
        social_manipulation_suspected=False,
    )
    assert result.approved is False
    assert result.final_decision == "NO TRADE"
    assert result.checklist["sufficient_liquidity"] is False


def test_contradictory_news_forces_no_trade():
    tf = make_filter()
    result = tf.approve(
        signal=make_buy_signal(), risk=make_risk(rr=2.67), avg_volume_20d=500_000, atr_pct_of_price=0.02,
        account_level_reasons=[], news_credible_sources=2, news_contradictory=True,
        social_manipulation_suspected=False,
    )
    assert result.approved is False


def test_manipulation_suspected_forces_no_trade():
    tf = make_filter()
    result = tf.approve(
        signal=make_buy_signal(), risk=make_risk(rr=2.67), avg_volume_20d=500_000, atr_pct_of_price=0.02,
        account_level_reasons=[], news_credible_sources=2, news_contradictory=False,
        social_manipulation_suspected=True,
    )
    assert result.approved is False


def test_account_level_rejection_forces_no_trade():
    tf = make_filter()
    result = tf.approve(
        signal=make_buy_signal(), risk=make_risk(rr=2.67), avg_volume_20d=500_000, atr_pct_of_price=0.02,
        account_level_reasons=["Daily loss limit reached."], news_credible_sources=2,
        news_contradictory=False, social_manipulation_suspected=False,
    )
    assert result.approved is False
    assert "Daily loss limit reached." in result.reasons


def test_insufficient_risk_reward_forces_no_trade():
    tf = make_filter(min_risk_reward=3.0)
    result = tf.approve(
        signal=make_buy_signal(), risk=make_risk(rr=2.0), avg_volume_20d=500_000, atr_pct_of_price=0.02,
        account_level_reasons=[], news_credible_sources=2, news_contradictory=False,
        social_manipulation_suspected=False,
    )
    assert result.approved is False
    assert result.checklist["risk_reward_sufficient"] is False


def test_missing_risk_assessment_forces_no_trade():
    tf = make_filter()
    result = tf.approve(
        signal=make_buy_signal(), risk=None, avg_volume_20d=500_000, atr_pct_of_price=0.02,
        account_level_reasons=[], news_credible_sources=2, news_contradictory=False,
        social_manipulation_suspected=False,
    )
    assert result.approved is False
    assert result.final_decision == "NO TRADE"


def test_no_news_available_does_not_block_an_otherwise_good_trade():
    """Absence of news (not fetched, or none found) must not be treated the
    same as 'news exists but is unreliable' -- otherwise every scan run
    with --no-news would be permanently unable to trade."""
    tf = make_filter()
    result = tf.approve(
        signal=make_buy_signal(), risk=make_risk(rr=2.67), avg_volume_20d=500_000, atr_pct_of_price=0.02,
        account_level_reasons=[], news_credible_sources=0, news_contradictory=False,
        social_manipulation_suspected=False, news_available=False,
    )
    assert result.checklist["news_credibility_acceptable"] is True
    assert result.approved is True


def test_news_found_but_none_credible_is_treated_as_unreliable():
    tf = make_filter()
    result = tf.approve(
        signal=make_buy_signal(), risk=make_risk(rr=2.67), avg_volume_20d=500_000, atr_pct_of_price=0.02,
        account_level_reasons=[], news_credible_sources=0, news_contradictory=False,
        social_manipulation_suspected=False, news_available=True,
    )
    assert result.checklist["news_credibility_acceptable"] is False
    assert result.approved is False


def make_buy_signal_without_market_condition(confidence=80.0, agreement=0.9) -> SignalDecision:
    """A signal where the benchmark/index trend could not be determined
    (SignalInputs.market_trend == 'UNKNOWN') -- signal_engine.py correctly
    EXCLUDES market_condition from component_scores in that case rather
    than fabricating a neutral score."""
    return SignalDecision(
        symbol="X",
        component_scores={"technical": 75, "fundamentals": 70, "news_sentiment": 68,
                           "social_sentiment": 60, "volume_price_behavior": 70, "risk_volatility": 65},
        overall_confidence=confidence, confidence_label="HIGH CONFIDENCE", direction="UP",
        model_agreement=agreement, decision="BUY", reasons=["ok"],
        unavailable_components=["market_condition"],
    )


def test_missing_market_condition_does_not_block_an_otherwise_good_trade():
    """Regression test: no benchmark/index data available (no index_symbol
    configured, index fetch failed, or insufficient index history) must not
    be treated the same as 'market conditions are bad'. Before this fix,
    `component_scores.get('market_condition', 0) >= 40` silently read back
    the absent key as 0 and failed unconditionally -- turning 'we don't have
    benchmark data' into an undisclosed mandatory gate that blocked every
    trade whenever the benchmark index was unavailable, exactly the failure
    mode spec section 9 forbids for optional/unavailable components."""
    tf = make_filter()
    result = tf.approve(
        signal=make_buy_signal_without_market_condition(), risk=make_risk(rr=2.67),
        avg_volume_20d=500_000, atr_pct_of_price=0.02, account_level_reasons=[],
        news_credible_sources=2, news_contradictory=False, social_manipulation_suspected=False,
        market_condition_available=False,
    )
    assert result.checklist["market_conditions_acceptable"] is True
    assert result.approved is True


def test_market_condition_available_but_poor_still_blocks_the_trade():
    """The flip side: when a benchmark trend WAS determined and it's simply
    unfavorable (score < 40), the check must still fail as before -- this
    fix only protects the 'genuinely unavailable' case, it does not weaken
    the check itself."""
    tf = make_filter()
    signal = make_buy_signal()
    signal.component_scores["market_condition"] = 20  # available, just poor
    result = tf.approve(
        signal=signal, risk=make_risk(rr=2.67), avg_volume_20d=500_000, atr_pct_of_price=0.02,
        account_level_reasons=[], news_credible_sources=2, news_contradictory=False,
        social_manipulation_suspected=False, market_condition_available=True,
    )
    assert result.checklist["market_conditions_acceptable"] is False
    assert result.approved is False


def test_hold_signal_never_gets_approved_as_a_trade():
    tf = make_filter()
    hold_signal = make_buy_signal()
    hold_signal.decision = "HOLD"
    result = tf.approve(
        signal=hold_signal, risk=make_risk(rr=2.67), avg_volume_20d=500_000, atr_pct_of_price=0.02,
        account_level_reasons=[], news_credible_sources=2, news_contradictory=False,
        social_manipulation_suspected=False,
    )
    assert result.final_decision == "NO TRADE"
