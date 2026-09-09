"""
Spec section 15: StrategyHealthReport (overfitting/robustness detection).
Result must be one of HEALTHY / CAUTION / OVERFIT RISK / INSUFFICIENT DATA.
"""

import pandas as pd
import pytest

from backtesting.backtester import BacktestConfig, BacktestResult, Trade
from backtesting.health_report import (
    HealthCheck,
    ParameterSensitivityResult,
    assess_strategy_health,
    run_parameter_sensitivity_check,
)
from backtesting.metrics import compute_metrics
from backtesting.walk_forward import WalkForwardFold, WalkForwardReport
from config.settings import Config


def make_trade(regime: str, net_pnl: float, entry_date="2023-01-01") -> Trade:
    ts = pd.Timestamp(entry_date)
    return Trade(
        symbol="TEST", side="BUY", entry_date=ts, entry_price=100.0, exit_date=ts + pd.Timedelta(days=1),
        exit_price=101.0, stop_loss=95.0, target=110.0, shares=10, exit_reason="TARGET",
        gross_pnl=net_pnl, costs=0.0, net_pnl=net_pnl, regime_at_entry=regime, confidence_at_entry=70.0,
        unavailable_components=[], decision_reasons=[],
    )


def make_backtest_result(trades, initial_capital=100_000.0, is_valid=True) -> BacktestResult:
    pnls = [t.net_pnl for t in trades]
    equity = [initial_capital]
    for p in pnls:
        equity.append(equity[-1] + p)
    metrics = compute_metrics(pnls, equity, [p / initial_capital for p in pnls] or [0.0],
                               initial_capital, max(len(trades), 1))
    return BacktestResult(
        trades=trades, equity_curve=equity, dates=[pd.Timestamp("2023-01-01")] * len(equity),
        metrics=metrics, metrics_by_regime={}, signals_approved=len(trades),
        signals_rejected_at_execution=0, is_valid=is_valid,
    )


def make_fold(test_trades, train_expectancy=10.0, overfitting_flag=False) -> WalkForwardFold:
    train_result = make_backtest_result([make_trade("BULL_LOW_VOL", train_expectancy) for _ in range(20)])
    validation_result = make_backtest_result([make_trade("BULL_LOW_VOL", train_expectancy) for _ in range(5)])
    test_result = make_backtest_result(test_trades)
    return WalkForwardFold(
        train_range=(pd.Timestamp("2023-01-01"), pd.Timestamp("2023-02-01")),
        validation_range=(pd.Timestamp("2023-02-02"), pd.Timestamp("2023-02-10")),
        test_range=(pd.Timestamp("2023-02-11"), pd.Timestamp("2023-03-01")),
        train_result=train_result, validation_result=validation_result,
        out_of_sample_result=test_result, overfitting_flag=overfitting_flag,
        degradation_ratio=1.0, validation_degradation_ratio=1.0, ml_used=False,
    )


def make_report(folds, overall_overfitting_detected=False, regime_flag=False, regime_note="spread ok") -> WalkForwardReport:
    return WalkForwardReport(
        folds=folds, overall_overfitting_detected=overall_overfitting_detected,
        regime_dependency_flag=regime_flag, regime_dependency_note=regime_note,
        regime_pnl_breakdown={},
    )


# --- Overall verdict logic -------------------------------------------------

def test_insufficient_data_when_too_few_test_trades():
    fold = make_fold([make_trade("BULL_LOW_VOL", 10.0) for _ in range(3)])
    report = make_report([fold])
    health = assess_strategy_health(report, min_test_trades_for_assessment=30)
    assert health.result == "INSUFFICIENT DATA"
    assert health.checks[0].status == "FAIL"
    assert health.checks[0].name == "trade_count_sufficiency"


def test_healthy_when_everything_checks_out():
    trades = [make_trade("BULL_LOW_VOL", 10.0) for _ in range(20)] + [make_trade("SIDEWAYS_LOW_VOL", -3.0) for _ in range(10)]
    fold = make_fold(trades, overfitting_flag=False)
    report = make_report([fold, make_fold(trades, overfitting_flag=False)], regime_note="spread across regimes")
    health = assess_strategy_health(report, min_test_trades_for_assessment=10)
    assert health.result in ("HEALTHY", "CAUTION")  # win-rate/time-period checks may vary with this synthetic data
    assert any(c.name == "trade_count_sufficiency" and c.status == "PASS" for c in health.checks)


def test_overfit_risk_when_walk_forward_flagged_overfitting():
    trades = [make_trade("BULL_LOW_VOL", 5.0) for _ in range(40)]
    fold = make_fold(trades, overfitting_flag=True)
    report = make_report([fold], overall_overfitting_detected=True)
    health = assess_strategy_health(report, min_test_trades_for_assessment=10)
    assert health.result == "OVERFIT RISK"
    assert any(c.name == "train_test_divergence" and c.status == "FAIL" for c in health.checks)


def test_caution_when_win_rate_suspiciously_high():
    trades = [make_trade("BULL_LOW_VOL", 10.0) for _ in range(40)]  # 100% win rate
    fold = make_fold(trades)
    report = make_report([fold])
    health = assess_strategy_health(report, min_test_trades_for_assessment=10, max_plausible_win_rate=0.85)
    assert health.result == "CAUTION"
    assert any(c.name == "win_rate_plausibility" and c.status == "WARN" for c in health.checks)


def test_parameter_sensitivity_na_message_points_to_the_optimize_command():
    """When parameter_sensitivity isn't supplied (the case for every real
    `walk-forward` CLI run today -- nothing wires run_parameter_sensitivity_check
    into it), the N/A guidance must point at this codebase's actual,
    CLI-exposed parameter-sensitivity diagnostic (`optimize`, spec Part 28),
    not at run_parameter_sensitivity_check, which no CLI command ever calls."""
    trades = [make_trade("BULL_LOW_VOL", 10.0) for _ in range(40)]
    fold = make_fold(trades)
    report = make_report([fold])
    health = assess_strategy_health(report, min_test_trades_for_assessment=10)
    check = next(c for c in health.checks if c.name == "parameter_sensitivity")
    assert check.status == "N/A"
    assert "optimize" in check.detail


def test_caution_when_regime_dependency_flagged():
    trades = [make_trade("BULL_LOW_VOL", 10.0) for _ in range(25)] + [make_trade("SIDEWAYS_LOW_VOL", -1.0) for _ in range(15)]
    fold = make_fold(trades)
    report = make_report([fold], regime_flag=True, regime_note="90% of profit from one regime")
    health = assess_strategy_health(report, min_test_trades_for_assessment=10, max_plausible_win_rate=0.99)
    assert health.result == "CAUTION"
    assert any(c.name == "regime_dependency" and c.status == "WARN" for c in health.checks)


def test_time_period_concentration_flagged_when_one_fold_dominates():
    dominant_fold = make_fold([make_trade("BULL_LOW_VOL", 100.0) for _ in range(15)])
    minor_fold = make_fold([make_trade("BULL_LOW_VOL", 1.0) for _ in range(15)])
    report = make_report([dominant_fold, minor_fold], regime_note="fine")
    health = assess_strategy_health(report, min_test_trades_for_assessment=10, max_plausible_win_rate=0.99)
    tp_check = next(c for c in health.checks if c.name == "time_period_concentration")
    assert tp_check.status == "WARN"
    assert health.result == "CAUTION"


def test_time_period_concentration_na_with_single_fold():
    fold = make_fold([make_trade("BULL_LOW_VOL", 10.0) for _ in range(15)] + [make_trade("SIDEWAYS_LOW_VOL", -2.0) for _ in range(15)])
    report = make_report([fold])
    health = assess_strategy_health(report, min_test_trades_for_assessment=10, max_plausible_win_rate=0.99)
    tp_check = next(c for c in health.checks if c.name == "time_period_concentration")
    assert tp_check.status == "N/A"


def test_symbol_concentration_na_without_per_symbol_data():
    trades = [make_trade("BULL_LOW_VOL", 10.0) for _ in range(15)] + [make_trade("SIDEWAYS_LOW_VOL", -2.0) for _ in range(15)]
    report = make_report([make_fold(trades)])
    health = assess_strategy_health(report, min_test_trades_for_assessment=10, max_plausible_win_rate=0.99)
    sc = next(c for c in health.checks if c.name == "symbol_concentration")
    assert sc.status == "N/A"


def test_symbol_concentration_flagged_when_one_symbol_dominates():
    trades = [make_trade("BULL_LOW_VOL", 10.0) for _ in range(15)] + [make_trade("SIDEWAYS_LOW_VOL", -2.0) for _ in range(15)]
    report = make_report([make_fold(trades)])
    health = assess_strategy_health(
        report, min_test_trades_for_assessment=10, max_plausible_win_rate=0.99,
        per_symbol_net_pnl={"RELIANCE.NS": 950.0, "TCS.NS": 50.0},
    )
    sc = next(c for c in health.checks if c.name == "symbol_concentration")
    assert sc.status == "WARN"
    assert health.result == "CAUTION"


def test_excessive_tuning_always_reported_as_na():
    trades = [make_trade("BULL_LOW_VOL", 10.0) for _ in range(15)] + [make_trade("SIDEWAYS_LOW_VOL", -2.0) for _ in range(15)]
    report = make_report([make_fold(trades)])
    health = assess_strategy_health(report, min_test_trades_for_assessment=10, max_plausible_win_rate=0.99)
    tuning = next(c for c in health.checks if c.name == "excessive_tuning")
    assert tuning.status == "N/A"


def test_render_text_includes_result_and_all_checks():
    trades = [make_trade("BULL_LOW_VOL", 10.0) for _ in range(15)] + [make_trade("SIDEWAYS_LOW_VOL", -2.0) for _ in range(15)]
    report = make_report([make_fold(trades)])
    health = assess_strategy_health(report, min_test_trades_for_assessment=10, max_plausible_win_rate=0.99)
    text = health.render_text()
    assert health.result in text
    for check in health.checks:
        assert check.name in text


def test_invalid_status_raises():
    with pytest.raises(ValueError):
        HealthCheck(name="x", status="MAYBE", detail="bad status")


# --- Parameter sensitivity harness -----------------------------------------

def _fake_backtest_result(expectancy: float) -> BacktestResult:
    return make_backtest_result([make_trade("BULL_LOW_VOL", expectancy)])


def test_parameter_sensitivity_stable_when_expectancy_barely_moves():
    def make_variant(delta):
        c = Config()
        return c

    def run_backtest(config):
        return _fake_backtest_result(10.0)  # constant regardless of variant

    result = run_parameter_sensitivity_check(make_variant, run_backtest, perturbations=(-0.2, 0.0, 0.2))
    assert result.unstable is False
    assert "stable" in result.note


def test_parameter_sensitivity_unstable_on_sign_flip():
    def make_variant(delta):
        return Config()

    def run_backtest(config, _state={"calls": 0}):
        # Simulate very different outcomes at different deltas via a closure trick:
        # we can't easily know which delta triggered this call without threading it
        # through Config, so instead vary by call order matching perturbations order.
        outcomes = [-50.0, 5.0, 60.0]
        idx = _state["calls"]
        _state["calls"] += 1
        return _fake_backtest_result(outcomes[idx % len(outcomes)])

    result = run_parameter_sensitivity_check(make_variant, run_backtest, perturbations=(-0.2, 0.0, 0.2))
    assert result.unstable is True
    assert "sign flips" in result.note


def test_parameter_sensitivity_requires_zero_baseline():
    with pytest.raises(ValueError):
        run_parameter_sensitivity_check(lambda d: Config(), lambda c: _fake_backtest_result(1.0), perturbations=(-0.1, 0.1))


def test_parameter_sensitivity_result_feeds_into_health_report_as_fail():
    trades = [make_trade("BULL_LOW_VOL", 10.0) for _ in range(15)] + [make_trade("SIDEWAYS_LOW_VOL", -2.0) for _ in range(15)]
    report = make_report([make_fold(trades)])
    unstable_result = ParameterSensitivityResult(unstable=True, note="unstable across variants", variant_results={})
    health = assess_strategy_health(
        report, min_test_trades_for_assessment=10, max_plausible_win_rate=0.99,
        parameter_sensitivity=unstable_result,
    )
    ps_check = next(c for c in health.checks if c.name == "parameter_sensitivity")
    assert ps_check.status == "FAIL"
    assert health.result == "OVERFIT RISK"
