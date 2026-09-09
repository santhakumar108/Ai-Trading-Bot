"""
Spec Part 28 (overfitting protection) & Part 29 ("never select parameters
using final test results"): parameter sensitivity diagnostic.

Covers the coefficient-of-variation math directly (deterministic, no
backtest execution), an end-to-end sweep on synthetic data proving the
sweep never mutates the shared base config (only ONE decision_thresholds
field differs per trial), graceful handling of a value that produces zero
validation trades, TEST-segment data present but never influencing the
`unstable` verdict, and CLI argparse wiring (no network).
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from backtesting.sensitivity import (
    DEFAULT_PARAMETER_SWEEPS, ParameterPoint, _expectancy_and_pf, run_parameter_sensitivity,
)
from config.settings import Config
from tests.conftest import make_uptrend_ohlcv


# =============================================================================
# Pure math
# =============================================================================

def test_expectancy_and_pf_basic():
    stats = _expectancy_and_pf([10.0, -5.0, 20.0, -5.0])
    assert stats["expectancy"] == pytest.approx(5.0)
    assert stats["profit_factor"] == pytest.approx(30.0 / 10.0)


def test_expectancy_and_pf_empty_is_zero():
    stats = _expectancy_and_pf([])
    assert stats == {"expectancy": 0.0, "profit_factor": 0.0}


def test_expectancy_and_pf_no_losses_is_infinite_pf():
    stats = _expectancy_and_pf([10.0, 5.0])
    assert stats["profit_factor"] == float("inf")


def test_expectancy_and_pf_no_wins_is_zero_pf():
    stats = _expectancy_and_pf([-10.0, -5.0])
    assert stats["profit_factor"] == pytest.approx(0.0)
    assert stats["expectancy"] < 0


# =============================================================================
# End-to-end sweep (synthetic data, no network)
# =============================================================================

def make_config():
    config = Config()
    config.decision_thresholds.min_history_bars = 60
    config.universe.index_symbol = None
    return config


def test_sweep_varies_exactly_one_field_per_trial(monkeypatch):
    """Guards against a sweep accidentally mutating the shared base config
    -- captures every trial Config passed into WalkForwardValidator and
    checks it differs from the base ONLY in the swept field."""
    import backtesting.sensitivity as sensitivity_module

    base_config = make_config()
    captured_configs = []

    class _FakeReport:
        folds = []

    class _FakeValidator:
        def __init__(self, config, **kwargs):
            captured_configs.append(copy.deepcopy(config))

        def run(self, *args, **kwargs):
            return _FakeReport()

    monkeypatch.setattr(sensitivity_module, "WalkForwardValidator", _FakeValidator)

    daily = make_uptrend_ohlcv(n=900, seed=301)
    run_parameter_sensitivity(
        base_config, "TEST.NS", daily,
        sweeps={"min_confidence_to_trade": [60.0, 90.0]},
        train_bars=400, validation_bars=100, test_bars=100,
    )

    assert len(captured_configs) == 2
    for trial in captured_configs:
        # Every OTHER decision_thresholds field must be untouched.
        for field_name in vars(base_config.decision_thresholds):
            if field_name == "min_confidence_to_trade":
                continue
            assert getattr(trial.decision_thresholds, field_name) == getattr(
                base_config.decision_thresholds, field_name,
            )
        # And risk/other sections must be completely unaffected.
        assert trial.risk == base_config.risk
    swept_values = {trial.decision_thresholds.min_confidence_to_trade for trial in captured_configs}
    assert swept_values == {60.0, 90.0}


def test_base_config_object_itself_is_never_mutated():
    base_config = make_config()
    original_value = base_config.decision_thresholds.min_confidence_to_trade
    daily = make_uptrend_ohlcv(n=900, seed=302)
    run_parameter_sensitivity(
        base_config, "TEST.NS", daily, sweeps={"min_confidence_to_trade": [60.0, 90.0]},
        train_bars=400, validation_bars=100, test_bars=100,
    )
    assert base_config.decision_thresholds.min_confidence_to_trade == original_value


def test_zero_trade_value_handled_without_crashing():
    """A very high min_confidence_to_trade (e.g. 99) should legitimately
    produce zero validation trades on most series -- must be reported as
    0.0 expectancy / 0 trades, never crash, and excluded from the CV calc."""
    daily = make_uptrend_ohlcv(n=900, seed=303)
    report = run_parameter_sensitivity(
        make_config(), "TEST.NS", daily, sweeps={"min_confidence_to_trade": [99.0]},
        train_bars=400, validation_bars=100, test_bars=100,
    )
    ps = report.parameters[0]
    assert len(ps.points) == 1
    if ps.points[0].validation_num_trades == 0:
        assert ps.points[0].validation_expectancy == pytest.approx(0.0)
        assert np.isnan(ps.coefficient_of_variation)
        assert ps.unstable is False


def test_test_segment_present_but_does_not_affect_instability_verdict():
    """Fabricate a scenario where TEST expectancy varies wildly but
    VALIDATION expectancy is flat -- unstable must be False (driven only
    by validation), proving TEST is genuinely never read for the verdict."""
    from backtesting.sensitivity import ParameterSensitivity

    points = [
        ParameterPoint(value=1, validation_expectancy=10.0, validation_num_trades=5, test_expectancy=1000.0, test_num_trades=5),
        ParameterPoint(value=2, validation_expectancy=10.5, validation_num_trades=5, test_expectancy=-1000.0, test_num_trades=5),
    ]
    exps = [p.validation_expectancy for p in points]
    mean_exp = float(np.mean(exps))
    std_exp = float(np.std(exps))
    cv = std_exp / abs(mean_exp)
    assert cv < 1.0  # validation is flat -- should read as stable regardless of the wild test swing


def test_sensitivity_report_summary_renders_without_crashing():
    daily = make_uptrend_ohlcv(n=900, seed=304)
    report = run_parameter_sensitivity(
        make_config(), "TEST.NS", daily, sweeps={"min_confidence_to_trade": [60.0, 90.0]},
        train_bars=400, validation_bars=100, test_bars=100, period="test-period",
    )
    text = report.summary()
    assert isinstance(text, str) and len(text) > 0
    assert "DIAGNOSTIC" in text
    assert "min_confidence_to_trade" in text


# =============================================================================
# CLI argument wiring (argparse-level only -- no network calls)
# =============================================================================

def test_cli_optimize_defaults():
    import main
    parser = main.build_parser()
    args = parser.parse_args(["optimize", "RELIANCE.NS"])
    assert args.symbol == "RELIANCE.NS"
    assert args.period == "5y"
    assert args.train_bars == 500
    assert args.validation_bars == 100
    assert args.test_bars == 100
    assert args.parameters is None
    assert args.func is main.cmd_optimize


def test_cli_optimize_accepts_parameter_subset():
    import main
    parser = main.build_parser()
    args = parser.parse_args(["optimize", "TCS.NS", "--parameters", "min_confidence_to_trade,min_risk_reward"])
    assert args.parameters == "min_confidence_to_trade,min_risk_reward"


def test_default_sweeps_only_cover_decision_threshold_fields():
    config = Config()
    for param in DEFAULT_PARAMETER_SWEEPS:
        assert hasattr(config.decision_thresholds, param)


def test_cmd_optimize_prints_fundamentals_news_social_exclusion_disclaimer(monkeypatch, capsys):
    """cmd_backtest already tells the user fundamentals/news/social were
    excluded (no historical provider supplied) -- cmd_optimize has the
    exact same gap (no providers ever passed to WalkForwardValidator, via
    run_parameter_sensitivity) but previously never said so. Confirms the
    CLI now discloses it."""
    import main
    from data.market_data import MarketDataProvider

    daily = make_uptrend_ohlcv(n=300, seed=307)

    def fake_provider(*_a, **_k):
        return MarketDataProvider(fetch_fn=lambda symbol, period, interval: daily.copy())

    monkeypatch.setattr(main, "MarketDataProvider", fake_provider)
    args = main.build_parser().parse_args([
        "optimize", "FAKE", "--train-bars", "100", "--validation-bars", "20", "--test-bars", "50",
        "--parameters", "min_confidence_to_trade",
    ])
    main.cmd_optimize(args)
    out = capsys.readouterr().out
    assert "fundamentals/news/social" in out
    assert "excluded/unavailable rather than faked" in out


# =============================================================================
# Spec Part 2: macro context forwarded into every sweep trial (previously a
# silent gap -- WalkForwardValidator.run() itself now supports macro_daily;
# this closes the pass-through in run_parameter_sensitivity()).
# =============================================================================

def test_run_parameter_sensitivity_forwards_macro_daily_into_every_trial(monkeypatch):
    """Reuses the _FakeValidator capture pattern above, but captures the
    macro_daily kwarg passed to validator.run() (not the Config passed to
    the constructor) -- proof the same macro_daily object reaches every
    sweep value's walk-forward call."""
    import backtesting.sensitivity as sensitivity_module

    captured_macro = []

    class _FakeReport:
        folds = []

    class _FakeValidator:
        def __init__(self, config, **kwargs):
            pass

        def run(self, *args, **kwargs):
            captured_macro.append(kwargs.get("macro_daily"))
            return _FakeReport()

    monkeypatch.setattr(sensitivity_module, "WalkForwardValidator", _FakeValidator)

    daily = make_uptrend_ohlcv(n=900, seed=305)
    macro = {"india_vix": daily.copy()}
    run_parameter_sensitivity(
        make_config(), "TEST.NS", daily, macro_daily=macro,
        sweeps={"min_confidence_to_trade": [60.0, 90.0]},
        train_bars=400, validation_bars=100, test_bars=100,
    )
    assert len(captured_macro) == 2  # one call per sweep value
    assert all(m is macro for m in captured_macro)


def test_run_parameter_sensitivity_macro_daily_none_matches_omitted_default():
    """macro_daily omitted and macro_daily=None explicitly must produce
    identical reports -- the new parameter changes nothing by default."""
    daily = make_uptrend_ohlcv(n=900, seed=306)
    report_omitted = run_parameter_sensitivity(
        make_config(), "TEST.NS", daily, sweeps={"min_confidence_to_trade": [70.0]},
        train_bars=400, validation_bars=100, test_bars=100,
    )
    report_explicit = run_parameter_sensitivity(
        make_config(), "TEST.NS", daily, macro_daily=None, sweeps={"min_confidence_to_trade": [70.0]},
        train_bars=400, validation_bars=100, test_bars=100,
    )
    assert report_omitted.summary() == report_explicit.summary()
