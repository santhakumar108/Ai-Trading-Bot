"""
Spec section 13 ("Walk-forward testing"): proper chronological
TRAIN -> VALIDATION -> TEST, never shuffled, reported separately.
Spec section 14 ("Regime-based analysis"): flag when out-of-sample
profitability depends on a single regime.
"""

import pandas as pd
import pytest

from backtesting.backtester import BacktestConfig, Trade
from backtesting.walk_forward import WalkForwardValidator, _regime_dependency_check
from config.settings import Config


def make_config(**overrides) -> Config:
    config = Config()
    config.decision_thresholds.min_history_bars = 60
    config.decision_thresholds.min_independent_signals = 1
    for k, v in overrides.items():
        if hasattr(config.decision_thresholds, k):
            setattr(config.decision_thresholds, k, v)
    return config


def make_trade(regime: str, net_pnl: float, entry_date="2023-01-01") -> Trade:
    ts = pd.Timestamp(entry_date)
    return Trade(
        symbol="TEST", side="BUY", entry_date=ts, entry_price=100.0, exit_date=ts + pd.Timedelta(days=1),
        exit_price=101.0, stop_loss=95.0, target=110.0, shares=10, exit_reason="TARGET",
        gross_pnl=net_pnl, costs=0.0, net_pnl=net_pnl, regime_at_entry=regime, confidence_at_entry=70.0,
        unavailable_components=[], decision_reasons=[],
    )


# --- Chronological, non-overlapping TRAIN -> VALIDATION -> TEST segments --

def test_walk_forward_runs_with_trade_outcome_ml_label_mode():
    """Spec Part 7: ml_label_mode='trade_outcome' is an opt-in alternative
    to the default direction-labeled ML model -- must run end-to-end
    without error and stay strictly point-in-time (ml_used only True when
    HistoricalMLProvider.fit_on_training_window actually returned a usable
    model for that fold's TRAIN window)."""
    from tests.conftest import make_synthetic_ohlcv

    daily = make_synthetic_ohlcv(n=700, drift=0.0015, volatility=0.01, seed=61)
    config = make_config()
    validator = WalkForwardValidator(
        config=config, backtest_config=BacktestConfig(initial_capital=100_000),
        ml_enabled=True, ml_label_mode="trade_outcome", ml_n_splits=3, ml_min_auc=0.0,
    )
    report = validator.run("TEST", daily, train_bars=300, validation_bars=50, test_bars=50, step_bars=100)
    assert len(report.folds) >= 1
    for fold in report.folds:
        assert isinstance(fold.ml_used, bool)


def test_fold_segments_are_chronological_and_non_overlapping(uptrend_daily):
    config = make_config()
    validator = WalkForwardValidator(config=config, backtest_config=BacktestConfig(initial_capital=100_000))
    report = validator.run("TEST", uptrend_daily, train_bars=100, validation_bars=20, test_bars=50, step_bars=50)
    assert len(report.folds) >= 1
    for fold in report.folds:
        train_end = fold.train_range[1]
        val_start, val_end = fold.validation_range
        test_start, test_end = fold.test_range
        assert train_end < val_start
        assert val_end < test_start
        assert val_start <= val_end
        assert test_start <= test_end


def test_fold_reports_train_validation_test_as_genuinely_separate_backtest_results(uptrend_daily):
    """train_result, validation_result, and out_of_sample_result must be
    three distinct BacktestResult objects (not the same run reused three
    times, and not the old combined train+validation 'in_sample_result')."""
    config = make_config()
    validator = WalkForwardValidator(config=config, backtest_config=BacktestConfig(initial_capital=100_000))
    report = validator.run("TEST", uptrend_daily, train_bars=100, validation_bars=20, test_bars=50, step_bars=50)
    for fold in report.folds:
        assert fold.train_result is not fold.validation_result
        assert fold.validation_result is not fold.out_of_sample_result
        assert len(fold.train_result.equity_curve) == 100
        assert len(fold.validation_result.equity_curve) == 100 + 20
        assert len(fold.out_of_sample_result.equity_curve) == 100 + 20 + 50


def test_validation_trades_are_gated_to_the_validation_window_only(uptrend_daily):
    config = make_config()
    validator = WalkForwardValidator(config=config, backtest_config=BacktestConfig(initial_capital=100_000))
    report = validator.run("TEST", uptrend_daily, train_bars=100, validation_bars=20, test_bars=50, step_bars=50)
    for fold in report.folds:
        val_start = fold.validation_range[0]
        for trade in fold.validation_result.trades:
            assert trade.entry_date >= val_start


def test_test_trades_are_gated_to_the_test_window_only(uptrend_daily):
    """Same guarantee as before, now against the renamed/clarified fields."""
    config = make_config()
    validator = WalkForwardValidator(config=config, backtest_config=BacktestConfig(initial_capital=100_000))
    report = validator.run("TEST", uptrend_daily, train_bars=100, validation_bars=20, test_bars=50, step_bars=50)
    for fold in report.folds:
        test_start = fold.test_range[0]
        for trade in fold.out_of_sample_result.trades:
            assert trade.entry_date >= test_start


def test_in_sample_result_alias_returns_train_result_for_backward_compatibility(uptrend_daily):
    config = make_config()
    validator = WalkForwardValidator(config=config, backtest_config=BacktestConfig(initial_capital=100_000))
    report = validator.run("TEST", uptrend_daily, train_bars=100, validation_bars=20, test_bars=50, step_bars=50)
    for fold in report.folds:
        assert fold.in_sample_result is fold.train_result


def test_degradation_ratios_are_computed_relative_to_train_expectancy(uptrend_daily):
    config = make_config()
    validator = WalkForwardValidator(config=config, backtest_config=BacktestConfig(initial_capital=100_000))
    report = validator.run("TEST", uptrend_daily, train_bars=100, validation_bars=20, test_bars=50, step_bars=50)
    for fold in report.folds:
        train_exp = fold.train_result.metrics.expectancy
        if train_exp not in (0, None):
            expected_val_ratio = fold.validation_result.metrics.expectancy / train_exp
            assert fold.validation_degradation_ratio == pytest.approx(expected_val_ratio)
            # `degradation_ratio` (test vs train) mirrors production's own
            # `min_test_trades` guard (backtesting/walk_forward.py): with too
            # few out-of-sample TEST trades to say anything meaningful, it is
            # deliberately left as NaN rather than computed from thin data --
            # that is not a bug, so this test must not assert a real ratio in
            # that case either.
            if fold.out_of_sample_result.metrics.num_trades >= validator.min_test_trades:
                expected_test_ratio = fold.out_of_sample_result.metrics.expectancy / train_exp
                assert fold.degradation_ratio == pytest.approx(expected_test_ratio)
            else:
                assert fold.degradation_ratio != fold.degradation_ratio  # NaN


def test_summary_mentions_train_validation_and_test(uptrend_daily):
    config = make_config()
    validator = WalkForwardValidator(config=config, backtest_config=BacktestConfig(initial_capital=100_000))
    report = validator.run("TEST", uptrend_daily, train_bars=100, validation_bars=20, test_bars=50, step_bars=50)
    text = report.summary()
    assert "TRAIN" in text and "VALIDATION" in text and "TEST" in text
    assert "REGIME DEPENDENCY" in text


# --- Regime dependency check (spec section 14) ----------------------------

def test_regime_dependency_flagged_when_one_regime_dominates_profit():
    trades = (
        [make_trade("BULL_LOW_VOL", 100.0) for _ in range(5)]
        + [make_trade("BEAR_HIGH_VOL", -10.0) for _ in range(3)]
        + [make_trade("BEAR_HIGH_VOL", 5.0) for _ in range(1)]
    )
    flagged, note, breakdown = _regime_dependency_check(trades, dominant_fraction_threshold=0.80)
    assert flagged is True
    assert "BULL_LOW_VOL" in note
    assert breakdown["BULL_LOW_VOL"] == pytest.approx(500.0)


def test_regime_dependency_not_flagged_when_profit_is_spread_across_regimes():
    trades = (
        [make_trade("BULL_LOW_VOL", 100.0) for _ in range(5)]
        + [make_trade("SIDEWAYS_LOW_VOL", 90.0) for _ in range(5)]
    )
    flagged, note, breakdown = _regime_dependency_check(trades, dominant_fraction_threshold=0.80)
    assert flagged is False


def test_regime_dependency_not_flagged_with_no_trades():
    flagged, note, breakdown = _regime_dependency_check([])
    assert flagged is False
    assert "No out-of-sample" in note
    assert breakdown == {}


def test_regime_dependency_not_flagged_when_only_one_regime_was_ever_traded():
    """A single-regime result is a data-insufficiency problem, not evidence
    of dependency either way -- must not be silently treated as 'fine'."""
    trades = [make_trade("BULL_LOW_VOL", 50.0) for _ in range(10)]
    flagged, note, breakdown = _regime_dependency_check(trades)
    assert flagged is False
    assert "single regime" in note
    assert "enough regime diversity" in note


def test_regime_dependency_not_flagged_when_net_unprofitable_everywhere():
    trades = [make_trade("BULL_LOW_VOL", -10.0) for _ in range(5)] + [make_trade("BEAR_HIGH_VOL", -20.0) for _ in range(5)]
    flagged, note, breakdown = _regime_dependency_check(trades)
    assert flagged is False
    assert "not net profitable" in note


def test_walk_forward_report_exposes_regime_pnl_breakdown_from_real_run(uptrend_daily):
    config = make_config()
    validator = WalkForwardValidator(config=config, backtest_config=BacktestConfig(initial_capital=100_000))
    report = validator.run("TEST", uptrend_daily, train_bars=100, validation_bars=20, test_bars=50, step_bars=50)
    assert isinstance(report.regime_pnl_breakdown, dict)
    assert isinstance(report.regime_dependency_note, str) and len(report.regime_dependency_note) > 0
    # breakdown must only reflect TEST-window trades, never train/validation
    all_test_trades = [t for f in report.folds for t in f.out_of_sample_result.trades]
    if all_test_trades:
        assert sum(report.regime_pnl_breakdown.values()) == pytest.approx(sum(t.net_pnl for t in all_test_trades))


# =============================================================================
# Spec Part 2: macro context wired into walk-forward (previously a silent
# macro_context=None gap -- Backtester.run() itself already supported
# macro_daily; this closes the pass-through in WalkForwardValidator.run()).
# =============================================================================

def test_walk_forward_macro_daily_none_matches_omitted_default(uptrend_daily):
    """`macro_daily` omitted entirely and `macro_daily=None` explicitly must
    produce byte-identical results -- the new parameter must not change
    default behavior for any existing caller."""
    config = make_config()
    validator = WalkForwardValidator(config=config, backtest_config=BacktestConfig(initial_capital=100_000))
    report_omitted = validator.run("TEST", uptrend_daily, train_bars=100, validation_bars=20, test_bars=50, step_bars=50)
    report_explicit = validator.run(
        "TEST", uptrend_daily, train_bars=100, validation_bars=20, test_bars=50, step_bars=50, macro_daily=None,
    )
    assert len(report_omitted.folds) == len(report_explicit.folds)
    for f1, f2 in zip(report_omitted.folds, report_explicit.folds):
        for seg1, seg2 in (
            (f1.train_result, f2.train_result),
            (f1.validation_result, f2.validation_result),
            (f1.out_of_sample_result, f2.out_of_sample_result),
        ):
            trades1 = [(t.entry_date, t.net_pnl) for t in seg1.trades]
            trades2 = [(t.entry_date, t.net_pnl) for t in seg2.trades]
            assert trades1 == trades2


def test_walk_forward_forwards_macro_daily_into_every_fold_segment(uptrend_daily, monkeypatch):
    """Capture-based proof that the SAME macro_daily dict passed into
    validator.run() reaches every internal Backtester.run() call (train,
    validation, AND out-of-sample) for every fold -- not just one segment."""
    from backtesting.backtester import Backtester

    captured = []
    original_run = Backtester.run

    def spy_run(self, *args, **kwargs):
        captured.append(kwargs.get("macro_daily"))
        return original_run(self, *args, **kwargs)

    monkeypatch.setattr(Backtester, "run", spy_run)

    config = make_config()
    validator = WalkForwardValidator(config=config, backtest_config=BacktestConfig(initial_capital=100_000))
    macro = {"india_vix": uptrend_daily.copy()}
    report = validator.run(
        "TEST", uptrend_daily, train_bars=100, validation_bars=20, test_bars=50, step_bars=50, macro_daily=macro,
    )
    assert len(report.folds) >= 1
    # 3 backtester.run() calls per fold: train, validation, out-of-sample.
    assert len(captured) == len(report.folds) * 3
    assert all(m is macro for m in captured)


def test_cmd_walk_forward_prints_fundamentals_news_social_exclusion_disclaimer(monkeypatch, capsys, uptrend_daily):
    """cmd_backtest already tells the user fundamentals/news/social were
    excluded (no historical provider supplied) -- cmd_walk_forward has the
    exact same gap (no providers ever passed to WalkForwardValidator) but
    previously never said so. Confirms the CLI now discloses it."""
    import main
    from data.market_data import MarketDataProvider

    def fake_provider(*_a, **_k):
        return MarketDataProvider(fetch_fn=lambda symbol, period, interval: uptrend_daily.copy())

    monkeypatch.setattr(main, "MarketDataProvider", fake_provider)
    args = main.build_parser().parse_args([
        "walk-forward", "FAKE", "--train-bars", "100", "--validation-bars", "20", "--test-bars", "50",
    ])
    main.cmd_walk_forward(args)
    out = capsys.readouterr().out
    assert "fundamentals/news/social" in out
    assert "excluded/unavailable rather than faked" in out


def test_walk_forward_macro_future_bars_never_affect_earlier_fold_trades():
    """Point-in-time safety at the walk-forward layer, reusing the exact
    corruption technique from test_macro_signal_integration.py's
    test_backtest_macro_future_bars_never_affect_earlier_trades: corrupt the
    macro series only AFTER a cutoff inside the (single) fold's TEST
    segment, and confirm every trade across train/validation/test entered
    with sufficient margin before that cutoff is byte-identical regardless
    of what happens to the macro series afterward."""
    from tests.conftest import make_downtrend_ohlcv, make_uptrend_ohlcv

    daily = make_uptrend_ohlcv(n=700, seed=601)
    india_vix_original = make_uptrend_ohlcv(n=700, seed=602)

    cutoff = 580  # inside the single fold's test segment (bars 500:600)
    india_vix_corrupted = india_vix_original.copy()
    replacement = make_downtrend_ohlcv(n=len(india_vix_original) - cutoff, seed=603)
    india_vix_corrupted.iloc[cutoff:] = replacement.values

    config = make_config()

    def run_once(macro_series):
        validator = WalkForwardValidator(config=config, backtest_config=BacktestConfig(initial_capital=100_000))
        # train+validation+test bars sum to 600 <= 700 (exactly one fold);
        # a step_bars larger than the remaining data means the loop stops
        # after that single fold, keeping this test deterministic.
        return validator.run(
            "TEST", daily, train_bars=400, validation_bars=100, test_bars=100, step_bars=999_999,
            macro_daily={"india_vix": macro_series},
        )

    report_original = run_once(india_vix_original)
    report_corrupted = run_once(india_vix_corrupted)
    assert len(report_original.folds) == 1
    assert len(report_corrupted.folds) == 1

    safe_cutoff_date = daily.index[cutoff - 60]
    for f_orig, f_corr in zip(report_original.folds, report_corrupted.folds):
        for seg_orig, seg_corr in (
            (f_orig.train_result, f_corr.train_result),
            (f_orig.validation_result, f_corr.validation_result),
            (f_orig.out_of_sample_result, f_corr.out_of_sample_result),
        ):
            early_orig = [t for t in seg_orig.trades if t.entry_date < safe_cutoff_date]
            early_corr = [t for t in seg_corr.trades if t.entry_date < safe_cutoff_date]
            assert len(early_orig) == len(early_corr)
            for a, b in zip(early_orig, early_corr):
                assert a.entry_date == b.entry_date
                assert a.entry_price == pytest.approx(b.entry_price)
                assert a.net_pnl == pytest.approx(b.net_pnl)
