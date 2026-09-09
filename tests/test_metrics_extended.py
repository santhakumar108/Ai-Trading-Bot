"""
Spec section 12 ("Performance analysis") additions: annualized volatility,
Calmar ratio, recovery factor, monthly/yearly return breakdowns, and the
NIFTY 50 (or configured benchmark) buy-and-hold comparison -- plus spec
sections 11 & 24(7)'s "document every execution assumption" requirement.
"""

import numpy as np
import pandas as pd
import pytest

from backtesting.backtester import Backtester, BacktestConfig, execution_assumptions
from backtesting.metrics import (
    PerformanceMetrics,
    annualized_volatility,
    compute_metrics,
    max_drawdown_currency,
    monthly_returns,
    yearly_returns,
)
from config.settings import Config
from tests.conftest import make_synthetic_ohlcv


def make_config(**overrides) -> Config:
    config = Config()
    config.decision_thresholds.min_history_bars = 60
    config.decision_thresholds.min_independent_signals = 1
    for k, v in overrides.items():
        if hasattr(config.decision_thresholds, k):
            setattr(config.decision_thresholds, k, v)
    return config


# --- annualized volatility ---------------------------------------------

def test_annualized_volatility_zero_for_constant_returns():
    assert annualized_volatility([0.001] * 30) == pytest.approx(0.0, abs=1e-9)


def test_annualized_volatility_positive_for_noisy_returns():
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0005, 0.02, size=200).tolist()
    vol = annualized_volatility(returns)
    assert vol > 0
    # sanity: annualized vol should be roughly daily_std * sqrt(252), same order of magnitude
    assert vol == pytest.approx(np.std(returns, ddof=1) * np.sqrt(252), rel=1e-6)


def test_annualized_volatility_needs_at_least_two_points():
    assert annualized_volatility([0.01]) == 0.0
    assert annualized_volatility([]) == 0.0


# --- Calmar ratio (via compute_metrics) ---------------------------------

def test_calmar_ratio_matches_cagr_over_abs_max_drawdown():
    pnls = [100, -50]
    equity = [100_000, 100_500, 100_100, 100_400, 99_000, 105_000]
    returns = [0.005, -0.004, 0.003, -0.014, 0.061]
    metrics = compute_metrics(pnls, equity, returns, initial_capital=100_000, num_trading_periods=5)
    expected = metrics.cagr_pct / abs(metrics.max_drawdown_pct) if metrics.max_drawdown_pct != 0 else 0.0
    assert metrics.calmar_ratio == pytest.approx(expected)


def test_calmar_ratio_is_zero_when_there_is_no_drawdown():
    equity = [100_000, 101_000, 102_000, 103_000]  # monotonically rising -> mdd == 0
    metrics = compute_metrics([100, 100, 100], equity, [0.01, 0.0099, 0.0098],
                               initial_capital=100_000, num_trading_periods=3)
    assert metrics.max_drawdown_pct == 0.0
    assert metrics.calmar_ratio == 0.0


# --- Recovery factor -----------------------------------------------------

def test_recovery_factor_is_net_profit_over_worst_drawdown_in_currency():
    equity = [100_000, 120_000, 90_000, 130_000]  # peak 120k -> trough 90k = 30k drawdown; net profit = 30k
    dd_currency = max_drawdown_currency(equity)
    assert dd_currency == pytest.approx(30_000)
    metrics = compute_metrics([20_000, -30_000, 40_000], equity, [0.2, -0.25, 0.44],
                               initial_capital=100_000, num_trading_periods=3)
    expected_recovery = (equity[-1] - 100_000) / dd_currency
    assert metrics.recovery_factor == pytest.approx(expected_recovery)


def test_recovery_factor_zero_when_no_profit_and_no_drawdown():
    equity = [100_000, 100_000, 100_000]
    metrics = compute_metrics([], equity, [0.0, 0.0], initial_capital=100_000, num_trading_periods=2)
    assert metrics.recovery_factor == 0.0


# --- Monthly / yearly returns --------------------------------------------

def test_monthly_returns_bucket_by_calendar_month_relative_to_prior_bucket():
    dates = pd.bdate_range("2023-01-02", periods=45)  # spans Jan, Feb, part of Mar
    equity = [100_000 + i * 100 for i in range(45)]  # steadily rising, deterministic
    result = monthly_returns(equity, list(dates), initial_capital=100_000)
    assert set(result.keys()) >= {"2023-01"}
    # Jan's return should be (Jan's last equity value / initial_capital) - 1
    jan_last_idx = max(i for i, d in enumerate(dates) if d.month == 1)
    expected_jan = (equity[jan_last_idx] - 100_000) / 100_000
    assert result["2023-01"] == pytest.approx(expected_jan)


def test_yearly_returns_bucket_by_calendar_year():
    dates = pd.bdate_range("2023-11-01", periods=80)  # crosses into 2024
    equity = [50_000 + i * 50 for i in range(80)]
    result = yearly_returns(equity, list(dates), initial_capital=50_000)
    assert "2023" in result and "2024" in result
    # 2024's return is relative to 2023's closing equity, not to initial_capital
    dec_31_idx = max(i for i, d in enumerate(dates) if d.year == 2023)
    final_idx = len(dates) - 1
    expected_2024 = (equity[final_idx] - equity[dec_31_idx]) / equity[dec_31_idx]
    assert result["2024"] == pytest.approx(expected_2024)


def test_monthly_returns_empty_when_dates_missing_or_mismatched():
    assert monthly_returns([100, 200], None, 100) == {}
    assert monthly_returns([100, 200], [pd.Timestamp("2023-01-01")], 100) == {}  # length mismatch
    assert monthly_returns([], [], 100) == {}


# --- as_dict rounds nested monthly/yearly dicts too ----------------------

def test_as_dict_rounds_floats_inside_monthly_and_yearly_dicts():
    metrics = PerformanceMetrics(
        total_return_pct=0.123456, cagr_pct=0.1, annualized_volatility_pct=0.2, win_rate_pct=0.5,
        average_win=1.0, average_loss=-1.0, profit_factor=1.5, max_drawdown_pct=-0.1,
        sharpe_ratio=1.0, sortino_ratio=1.0, calmar_ratio=1.0, expectancy=0.0, num_trades=2,
        longest_losing_streak=1, recovery_factor=2.0, risk_adjusted_return=1.0,
        monthly_returns={"2023-01": 0.123456789}, yearly_returns={"2023": 0.987654321},
    )
    out = metrics.as_dict()
    assert out["monthly_returns"]["2023-01"] == pytest.approx(0.1235, abs=1e-4)
    assert out["yearly_returns"]["2023"] == pytest.approx(0.9877, abs=1e-4)
    assert out["total_return_pct"] == pytest.approx(0.1235, abs=1e-4)


# --- execution_assumptions() ----------------------------------------------

def test_execution_assumptions_documents_every_major_assumption():
    config = make_config()
    bt_config = BacktestConfig()
    assumptions = execution_assumptions(config, bt_config)
    expected_keys = {
        "fill_timing", "gaps", "exit_priority", "timeout_exit", "slippage_and_spread",
        "transaction_costs", "partial_fills", "position_sizing", "daily_and_weekly_loss_limits",
        "concurrent_positions_and_sector_exposure", "trading_calendar_and_market_hours",
        "data_quality_gate", "corporate_actions",
    }
    assert expected_keys <= set(assumptions.keys())
    for key, description in assumptions.items():
        assert isinstance(description, str) and len(description) > 20, key


# --- Benchmark comparison, via Backtester.run() ---------------------------

def test_benchmark_comparison_available_when_index_daily_supplied(uptrend_daily):
    config = make_config()
    bt = Backtester(config=config, backtest_config=BacktestConfig(initial_capital=100_000))
    index_daily = make_synthetic_ohlcv(n=len(uptrend_daily), drift=0.0005, volatility=0.01, seed=99,
                                        start_date=str(uptrend_daily.index[0].date()))
    result = bt.run("TEST", uptrend_daily, index_daily=index_daily, index_symbol="^NSEI")
    bc = result.benchmark_comparison
    assert bc is not None
    assert bc.benchmark_available is True
    assert bc.symbol == "^NSEI"
    expected_buy_hold = index_daily["Close"].iloc[-1] / index_daily["Close"].iloc[0] - 1
    assert bc.buy_hold_return_pct == pytest.approx(expected_buy_hold, rel=1e-6)
    assert bc.strategy_return_pct == pytest.approx(result.metrics.total_return_pct)
    assert bc.outperformance_pct == pytest.approx(bc.strategy_return_pct - bc.buy_hold_return_pct)


def test_benchmark_comparison_unavailable_without_index_daily(uptrend_daily):
    config = make_config()
    bt = Backtester(config=config)
    result = bt.run("TEST", uptrend_daily)
    assert result.benchmark_comparison is not None
    assert result.benchmark_comparison.benchmark_available is False
    assert result.benchmark_comparison.note  # a real explanation, not silently empty


def test_benchmark_comparison_marked_unavailable_when_run_is_data_quality_invalid(uptrend_daily):
    config = make_config()
    corrupted = pd.concat([uptrend_daily.iloc[:5], uptrend_daily.iloc[2:3], uptrend_daily.iloc[5:]])
    bt = Backtester(config=config)
    result = bt.run("TEST", corrupted)
    assert result.is_valid is False
    assert result.benchmark_comparison.benchmark_available is False


def test_backtest_result_carries_assumptions_even_on_invalid_run(uptrend_daily):
    config = make_config()
    corrupted = pd.concat([uptrend_daily.iloc[:5], uptrend_daily.iloc[2:3], uptrend_daily.iloc[5:]])
    bt = Backtester(config=config)
    result = bt.run("TEST", corrupted)
    assert "fill_timing" in result.assumptions


def test_overall_metrics_include_monthly_returns_from_a_real_backtest_run(uptrend_daily):
    config = make_config()
    bt = Backtester(config=config)
    result = bt.run("TEST", uptrend_daily)
    # A 300-bar run starting 2023-01-02 spans several calendar months regardless of trade activity.
    assert len(result.metrics.monthly_returns) >= 3
    assert len(result.metrics.yearly_returns) >= 1


def test_regime_metrics_do_not_include_monthly_yearly_breakdowns(uptrend_daily):
    """Per-regime metrics are built from a synthetic cumulative-P&L series
    with gaps between entries, not a real daily equity curve -- a
    monthly/yearly breakdown of that would be meaningless, so it must stay
    empty there even when the overall run's metrics have one."""
    config = make_config()
    bt = Backtester(config=config)
    result = bt.run("TEST", uptrend_daily)
    for regime_metrics in result.metrics_by_regime.values():
        assert regime_metrics.monthly_returns == {}
        assert regime_metrics.yearly_returns == {}
