"""
Spec section 16: Monte Carlo / robustness testing, using historical trade
results only -- never fabricated -- and never presented as a prediction of
future profit (checked directly via the disclaimer text).
"""

import numpy as np
import pytest

from backtesting.monte_carlo import run_monte_carlo


def test_empty_trades_marks_insufficient_data_and_does_not_crash():
    report = run_monte_carlo([], initial_capital=100_000, num_simulations=100, seed=1)
    assert report.insufficient_data is True
    assert report.num_simulations == 0
    assert report.actual_total_return_pct == 0.0


def test_few_trades_flagged_as_insufficient_but_still_computed():
    pnls = [100, -50, 200]
    report = run_monte_carlo(pnls, initial_capital=100_000, num_simulations=200, seed=1,
                              min_trades_recommended=20)
    assert report.insufficient_data is True
    assert report.num_simulations == 200
    assert len(report.total_return_pct_distribution) == 200


def test_bootstrap_distribution_centers_near_actual_for_iid_like_trades():
    rng = np.random.default_rng(0)
    pnls = rng.normal(loc=50, scale=20, size=200).tolist()
    report = run_monte_carlo(pnls, initial_capital=100_000, num_simulations=1000, seed=2, method="bootstrap")
    assert report.method == "bootstrap"
    median_return = report.percentiles[50]["total_return_pct"]
    assert median_return == pytest.approx(report.actual_total_return_pct, rel=0.5)  # same ballpark, not exact


def test_shuffle_uses_the_exact_same_trades_every_time():
    pnls = [100, -200, 50, -30, 400, -10]
    report = run_monte_carlo(pnls, initial_capital=10_000, num_simulations=500, seed=3, method="shuffle")
    # total P&L must be identical across every shuffle (same trades, different order)
    total_pnl = sum(pnls)
    for final_equity in report.final_equity_distribution:
        assert final_equity == pytest.approx(10_000 + total_pnl)
    # but drawdown/streak should vary with order
    assert len(set(round(d, 6) for d in report.max_drawdown_pct_distribution)) > 1


def test_shuffle_forces_trades_per_simulation_to_input_length():
    pnls = [10, -5, 20, -15, 30]
    report = run_monte_carlo(pnls, initial_capital=50_000, num_simulations=50, seed=4, method="shuffle",
                              trades_per_simulation=999)  # should be overridden, not honored
    assert report.trades_per_simulation == len(pnls)


def test_invalid_method_raises():
    with pytest.raises(ValueError):
        run_monte_carlo([1, 2, 3], initial_capital=1000, method="magic")


def test_probability_of_ruin_is_one_when_a_single_huge_loss_wipes_out_capital():
    pnls = [-999_999] * 10
    report = run_monte_carlo(pnls, initial_capital=100, num_simulations=100, seed=5, method="shuffle")
    assert report.probability_of_ruin == pytest.approx(1.0)


def test_probability_of_ruin_is_zero_when_no_trade_can_breach_capital():
    pnls = [10, -5, 20, -8, 15] * 4
    report = run_monte_carlo(pnls, initial_capital=1_000_000, num_simulations=200, seed=6)
    assert report.probability_of_ruin == pytest.approx(0.0)


def test_probability_of_net_loss_is_zero_when_every_trade_is_a_win():
    pnls = [10, 20, 30, 40, 50] * 5
    report = run_monte_carlo(pnls, initial_capital=100_000, num_simulations=300, seed=7)
    assert report.probability_of_net_loss == pytest.approx(0.0)


def test_probability_of_net_loss_is_high_when_every_trade_is_a_loss():
    pnls = [-10, -20, -30, -40, -50] * 5
    report = run_monte_carlo(pnls, initial_capital=100_000, num_simulations=300, seed=8)
    assert report.probability_of_net_loss == pytest.approx(1.0)


def test_probability_drawdown_worse_than_matches_manual_count():
    pnls = [10, -50, 20, -80, 30, -10] * 5
    report = run_monte_carlo(pnls, initial_capital=100_000, num_simulations=500, seed=9, method="shuffle")
    threshold = 0.001
    expected = sum(1 for d in report.max_drawdown_pct_distribution if d <= -threshold) / len(report.max_drawdown_pct_distribution)
    assert report.probability_drawdown_worse_than(threshold) == pytest.approx(expected)


def test_probability_drawdown_worse_than_nan_with_no_simulations():
    report = run_monte_carlo([], initial_capital=100_000)
    assert report.probability_drawdown_worse_than(0.1) != report.probability_drawdown_worse_than(0.1)  # NaN != NaN


def test_percentiles_are_monotonic_for_total_return():
    rng = np.random.default_rng(11)
    pnls = rng.normal(20, 30, size=100).tolist()
    report = run_monte_carlo(pnls, initial_capital=100_000, num_simulations=800, seed=12,
                              percentiles=(5, 25, 50, 75, 95))
    values = [report.percentiles[p]["total_return_pct"] for p in (5, 25, 50, 75, 95)]
    assert values == sorted(values)


def test_summary_includes_disclaimer_language_about_not_being_a_prediction():
    report = run_monte_carlo([10, -5, 20], initial_capital=10_000, num_simulations=50, seed=13)
    text = report.summary()
    assert "NOT a prediction" in text or "not a prediction" in text.lower()


def test_disclaimer_field_is_never_empty():
    for method in ("bootstrap", "shuffle"):
        report = run_monte_carlo([10, -5, 20, -8, 15], initial_capital=10_000, num_simulations=20,
                                  seed=14, method=method)
        assert len(report.disclaimer) > 50


def test_actual_metrics_reflect_the_real_historical_order_not_a_simulation():
    pnls = [100, -300, 50]  # equity: 1000 -> 1100 -> 800 -> 850; mdd from peak 1100 to trough 800
    report = run_monte_carlo(pnls, initial_capital=1000, num_simulations=10, seed=15, method="shuffle")
    assert report.actual_total_return_pct == pytest.approx((850 - 1000) / 1000)
    assert report.actual_max_drawdown_pct == pytest.approx((800 - 1100) / 1100)
    assert report.actual_longest_losing_streak == 1
