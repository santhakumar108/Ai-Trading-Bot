from datetime import date, timedelta

import pytest

from risk.risk_engine import CapitalProtection, TradeRiskCalculator


def make_calculator(**overrides):
    defaults = dict(
        risk_per_trade_pct=0.01,
        min_risk_reward=2.0,
        transaction_cost_pct=0.001,
        slippage_pct=0.0007,
        min_edge_after_costs_pct=0.001,
        max_single_position_pct=0.20,
    )
    defaults.update(overrides)
    return TradeRiskCalculator(**defaults)


def test_example_from_spec_is_approved_and_matches_math():
    """Entry=100, Stop=97, Target=108 -> Risk=3, Reward=8, RR=2.67 (spec example)."""
    calc = make_calculator(transaction_cost_pct=0.0, slippage_pct=0.0)
    result = calc.evaluate("TEST", "BUY", entry=100, stop_loss=97, target=108, capital=1_000_000)
    assert result.risk_per_share == pytest.approx(3.0)
    assert result.reward_per_share == pytest.approx(8.0)
    assert result.risk_reward_ratio == pytest.approx(8 / 3, rel=1e-6)
    assert result.approved is True
    assert result.position_size > 0


def test_rejects_when_risk_reward_below_minimum():
    calc = make_calculator()
    result = calc.evaluate("TEST", "BUY", entry=100, stop_loss=97, target=103, capital=1_000_000)  # RR = 1.0
    assert result.approved is False
    assert any("Risk/reward" in r for r in result.rejection_reasons)


def test_rejects_illogical_stop_loss_for_buy():
    calc = make_calculator()
    # Stop above entry on a BUY is not logically a stop-loss.
    result = calc.evaluate("TEST", "BUY", entry=100, stop_loss=105, target=120, capital=1_000_000)
    assert result.approved is False
    assert any("logically defined" in r for r in result.rejection_reasons)


def test_position_size_is_capital_and_stop_driven_not_confidence_driven():
    """Position sizing must depend only on capital/risk%/entry/stop -- calling
    evaluate() twice with identical inputs must give identical sizing,
    regardless of any 'confidence' the caller might have computed elsewhere
    (this module doesn't even accept a confidence parameter, by design)."""
    calc = make_calculator()
    r1 = calc.evaluate("TEST", "BUY", entry=50, stop_loss=48, target=56, capital=500_000)
    r2 = calc.evaluate("TEST", "BUY", entry=50, stop_loss=48, target=56, capital=500_000)
    assert r1.position_size == r2.position_size


def test_costs_can_flip_a_marginal_trade_to_rejected():
    calc = make_calculator(transaction_cost_pct=0.05, slippage_pct=0.05, min_edge_after_costs_pct=0.001)
    result = calc.evaluate("TEST", "BUY", entry=100, stop_loss=97, target=108, capital=1_000_000)
    assert result.approved is False
    assert result.net_risk_reward_ratio < result.risk_reward_ratio


def test_sell_side_math():
    calc = make_calculator(transaction_cost_pct=0.0, slippage_pct=0.0)
    result = calc.evaluate("TEST", "SELL", entry=100, stop_loss=104, target=90, capital=1_000_000)
    assert result.risk_per_share == pytest.approx(4.0)
    assert result.reward_per_share == pytest.approx(10.0)
    assert result.approved is True


# --- Expected value gate (spec section 10) ---------------------------------

def test_expected_value_uses_conservative_default_win_probability_of_half():
    calc = make_calculator(transaction_cost_pct=0.0, slippage_pct=0.0)
    result = calc.evaluate("TEST", "BUY", entry=100, stop_loss=97, target=108, capital=1_000_000)
    assert result.assumed_win_probability == pytest.approx(0.5)
    # EV = 0.5*8 - 0.5*3 = 2.5 at zero cost
    assert result.expected_value_per_share == pytest.approx(2.5)
    assert result.expected_value_positive is True


def test_expected_value_gate_rejects_a_trade_below_the_configured_minimum():
    calc = make_calculator(transaction_cost_pct=0.0, slippage_pct=0.0, min_expected_value_per_share=100.0)
    result = calc.evaluate("TEST", "BUY", entry=100, stop_loss=97, target=108, capital=1_000_000)
    assert result.expected_value_positive is False
    assert result.approved is False
    assert any("Expected value" in r for r in result.rejection_reasons)


def test_expected_value_charges_costs_on_the_losing_leg_too():
    """Unlike net_risk_reward_ratio (which only nets costs off the reward),
    expected value must also charge the round-trip cost on a LOSING trade --
    a stopped-out position still pays brokerage and slippage."""
    cheap = make_calculator(transaction_cost_pct=0.0, slippage_pct=0.0)
    costly = make_calculator(transaction_cost_pct=0.02, slippage_pct=0.01)
    cheap_result = cheap.evaluate("TEST", "BUY", entry=100, stop_loss=97, target=108, capital=1_000_000)
    costly_result = costly.evaluate("TEST", "BUY", entry=100, stop_loss=97, target=108, capital=1_000_000)
    assert costly_result.expected_value_per_share < cheap_result.expected_value_per_share


def test_win_probability_fn_hook_overrides_the_flat_default_when_supplied():
    calc = make_calculator(
        transaction_cost_pct=0.0, slippage_pct=0.0,
        win_probability_fn=lambda confidence: 0.9 if confidence >= 80 else 0.5,
    )
    high_conf = calc.evaluate("TEST", "BUY", entry=100, stop_loss=97, target=108, capital=1_000_000, confidence=90)
    low_conf = calc.evaluate("TEST", "BUY", entry=100, stop_loss=97, target=108, capital=1_000_000, confidence=50)
    assert high_conf.assumed_win_probability == pytest.approx(0.9)
    assert low_conf.assumed_win_probability == pytest.approx(0.5)
    assert high_conf.expected_value_per_share > low_conf.expected_value_per_share


def test_win_probability_fn_ignored_without_a_confidence_argument():
    calc = make_calculator(
        transaction_cost_pct=0.0, slippage_pct=0.0, win_probability_fn=lambda c: 0.99,
    )
    result = calc.evaluate("TEST", "BUY", entry=100, stop_loss=97, target=108, capital=1_000_000)  # no confidence
    assert result.assumed_win_probability == pytest.approx(0.5)  # falls back to the flat default


# --- Capital protection -----------------------------------------------------

def make_protection(**overrides):
    defaults = dict(
        starting_capital=1_000_000.0,
        max_daily_loss_pct=0.02,
        max_weekly_loss_pct=0.05,
        max_simultaneous_positions=3,
        max_sector_exposure_pct=0.30,
        max_single_position_pct=0.20,
        today=date(2024, 1, 2),
    )
    defaults.update(overrides)
    return CapitalProtection(**defaults)


def test_daily_loss_limit_halts_new_trades():
    protection = make_protection()
    protection.register_close(sector=None, notional_exposure=100_000, realized_pnl=-25_000, today=date(2024, 1, 2))
    reasons = protection.pre_trade_check("AAPL", None, 10_000, today=date(2024, 1, 2))
    assert any("Daily loss limit" in r or "Trading halted" in r for r in reasons)


def test_daily_loss_limit_resets_next_day():
    protection = make_protection()
    protection.register_close(sector=None, notional_exposure=100_000, realized_pnl=-25_000, today=date(2024, 1, 2))
    protection.resume()
    reasons = protection.pre_trade_check("AAPL", None, 10_000, today=date(2024, 1, 3))
    assert reasons == []


def test_max_simultaneous_positions_enforced():
    protection = make_protection(max_simultaneous_positions=1)
    protection.register_open(sector=None, notional_exposure=10_000)
    reasons = protection.pre_trade_check("AAPL", None, 10_000, today=date(2024, 1, 2))
    assert any("Max simultaneous positions" in r for r in reasons)


def test_sector_exposure_limit_enforced():
    protection = make_protection(max_sector_exposure_pct=0.05)
    reasons = protection.pre_trade_check("AAPL", "TECH", 100_000, today=date(2024, 1, 2))
    assert any("Sector exposure" in r for r in reasons)


def test_emergency_stop_blocks_everything():
    protection = make_protection()
    protection.emergency_stop("Manual test stop.")
    reasons = protection.pre_trade_check("AAPL", None, 1_000, today=date(2024, 1, 2))
    assert any("Trading halted" in r for r in reasons)


def test_abnormal_market_move_halts_trading():
    protection = make_protection()
    triggered = protection.detect_abnormal_market(index_daily_return=-0.07, threshold=0.05)
    assert triggered is True
    assert protection.state.trading_halted is True
