"""
strategy/report.py: TradeReport / build_trade_report, including the
Expected Value fields added for spec sections 10 & 18 (per-candidate
Expected Value must be surfaced, never fabricated when there's no risk
assessment to compute it from).
"""

import pytest

from risk.risk_engine import TradeRiskCalculator
from strategy.report import build_trade_report
from strategy.signal_engine import SignalDecision
from strategy.trade_filter import FilterResult


def make_signal(decision="BUY", confidence=80.0) -> SignalDecision:
    return SignalDecision(
        symbol="TEST", component_scores={"technical": 70.0}, overall_confidence=confidence,
        confidence_label="HIGH", direction="UP" if decision == "BUY" else "NONE",
        model_agreement=1.0, decision=decision, reasons=["strong uptrend"],
    )


def make_filter_result(approved=True, decision="BUY") -> FilterResult:
    return FilterResult(
        symbol="TEST", approved=approved, final_decision=decision,
        checklist={"signal_is_directional": True}, reasons=["clean setup"],
    )


def make_risk_calculator(**overrides):
    defaults = dict(
        risk_per_trade_pct=0.01, min_risk_reward=2.0, transaction_cost_pct=0.0,
        slippage_pct=0.0, min_edge_after_costs_pct=0.0, max_single_position_pct=0.20,
    )
    defaults.update(overrides)
    return TradeRiskCalculator(**defaults)


def test_expected_value_populated_when_risk_assessment_present():
    calc = make_risk_calculator(default_win_probability=0.6)
    risk = calc.evaluate("TEST", "BUY", entry=100, stop_loss=97, target=108, capital=1_000_000)
    report = build_trade_report(
        signal=make_signal(), filter_result=make_filter_result(), current_price=100,
        market_trend="UPTREND", risk=risk,
    )
    assert report.expected_value_per_share == pytest.approx(risk.expected_value_per_share)
    assert report.expected_value_total == pytest.approx(risk.expected_value_total)
    assert report.expected_value_per_share is not None


def test_expected_value_is_none_without_a_risk_assessment():
    """HOLD/NO TRADE decisions have no risk assessment -- EV must show as
    None (rendered N/A), never a fabricated 0.0."""
    report = build_trade_report(
        signal=make_signal(decision="NO TRADE", confidence=0.0),
        filter_result=make_filter_result(approved=False, decision="NO TRADE"),
        current_price=100, market_trend="SIDEWAYS", risk=None,
    )
    assert report.expected_value_per_share is None
    assert report.expected_value_total is None


def test_render_text_includes_expected_value_lines():
    calc = make_risk_calculator()
    risk = calc.evaluate("TEST", "BUY", entry=100, stop_loss=97, target=108, capital=1_000_000)
    report = build_trade_report(
        signal=make_signal(), filter_result=make_filter_result(), current_price=100,
        market_trend="UPTREND", risk=risk,
    )
    text = report.render_text()
    assert "Expected Value (per share):" in text
    assert "Expected Value (position):" in text
    assert "N/A" not in text.split("Expected Value (per share):")[1].split("\n")[0]


def test_render_text_shows_na_for_expected_value_when_missing():
    report = build_trade_report(
        signal=make_signal(decision="NO TRADE", confidence=0.0),
        filter_result=make_filter_result(approved=False, decision="NO TRADE"),
        current_price=100, market_trend="SIDEWAYS", risk=None,
    )
    text = report.render_text()
    assert "Expected Value (per share): N/A" in text
    assert "Expected Value (position): N/A" in text


def test_expected_value_total_equals_per_share_times_position_size():
    calc = make_risk_calculator()
    risk = calc.evaluate("TEST", "BUY", entry=100, stop_loss=97, target=108, capital=1_000_000)
    report = build_trade_report(
        signal=make_signal(), filter_result=make_filter_result(), current_price=100,
        market_trend="UPTREND", risk=risk,
    )
    assert report.expected_value_total == pytest.approx(report.expected_value_per_share * risk.position_size)
