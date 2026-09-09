"""
Walk-Forward Validation.

Splits historical data into successive (TRAIN, VALIDATION, TEST) windows
that always move strictly forward in time -- never shuffled (spec section
13: "never shuffle time-series data"). For each fold, the SAME `Backtester`
/ `StrategyPipeline` (see `backtesting/backtester.py`, `strategy/pipeline.py`)
is run three times, over three genuinely separate segments:

  * TRAIN: the fitting window itself. If `ml_enabled=True`, the
    `MLBaselineModel` is fit ONLY on this window (never on validation or
    test bars) via `HistoricalMLProvider.fit_on_training_window`.
  * VALIDATION: the FIRST held-out segment, immediately after train. The
    pipeline sees the full point-in-time history leading into it (exactly
    as a live system would), but only bars inside the validation window are
    allowed to open a new position (`trade_from`). This is a genuine
    out-of-sample check on the train-fitted model/strategy, reported
    separately from train (spec section 13: "report in-sample/validation/
    out-of-sample separately").
  * TEST: the SECOND, final held-out segment, after validation. Same
    point-in-time-history-but-gated-entries mechanism as validation. This
    is the number that matters most for "did this survive walk-forward,"
    precisely because nothing about the strategy or model was ever fit,
    tuned, or selected against it.

Overfitting detection (train -> test expectancy collapse): if out-of-sample
(test) expectancy collapses relative to train expectancy (below
`degradation_threshold` of it, or flips sign), the fold is flagged. Per the
spec, such a strategy "must be rejected", not shipped with a caveat. A
train -> validation degradation ratio is reported too, for the same reason,
though it does not by itself set `overfitting_flag` (see
`backtesting/health_report.py`, spec section 15, for a fuller overfitting
assessment that combines this with other signals).

Regime dependency (spec section 14): `WalkForwardReport` aggregates net P&L
by regime across every fold's TEST-window trades ONLY (never train/validation
trades, which could look falsely diversified from a strategy that was
effectively selected to work on the training regimes) and flags when one
regime accounts for the large majority of total out-of-sample profit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

from backtesting.backtester import Backtester, BacktestConfig, BacktestResult, Trade
from backtesting.historical_providers import (
    HistoricalFundamentalsProvider,
    HistoricalMLProvider,
    HistoricalNewsProvider,
    HistoricalSocialProvider,
)
from config.settings import Config


def _regime_dependency_check(
    trades: List[Trade], dominant_fraction_threshold: float = 0.80,
) -> Tuple[bool, str, Dict[str, float]]:
    """Spec section 14: 'flag if performance only works in one regime.'
    Looks ONLY at the trades passed in (callers pass TEST-window trades
    across all folds, never train/validation) so this reflects genuine
    out-of-sample regime dependency, not a strategy that was simply
    exposed to a narrow set of regimes during fitting."""
    breakdown: Dict[str, float] = {}
    for t in trades:
        breakdown[t.regime_at_entry] = breakdown.get(t.regime_at_entry, 0.0) + t.net_pnl

    if not trades:
        return False, "No out-of-sample (test-window) trades were taken across any fold -- cannot assess regime dependency.", breakdown

    distinct_regimes = {t.regime_at_entry for t in trades}
    if len(distinct_regimes) < 2:
        only = next(iter(distinct_regimes))
        return False, (
            f"All out-of-sample trades occurred in a single regime ({only}) -- there isn't enough "
            "regime diversity in this data/period to assess dependency either way; more history "
            "covering different regimes is needed before this can be judged."
        ), breakdown

    regimes_with_profit = {r: v for r, v in breakdown.items() if v > 0}
    total_profit = sum(regimes_with_profit.values())
    if total_profit <= 0:
        return False, (
            "Out-of-sample trading was not net profitable in any regime; regime dependency is moot "
            "until the strategy is profitable out-of-sample at all."
        ), breakdown

    dominant_regime, dominant_profit = max(regimes_with_profit.items(), key=lambda kv: kv[1])
    dominant_share = dominant_profit / total_profit
    if dominant_share >= dominant_fraction_threshold:
        return True, (
            f"{dominant_share:.0%} of total out-of-sample net profit came from a single regime "
            f"({dominant_regime}) across {len(distinct_regimes)} regime(s) traded -- this strategy's "
            "edge may not generalize outside that regime; do not extrapolate this result to all "
            "market conditions."
        ), breakdown

    return False, (
        f"Out-of-sample profit is spread across {len(regimes_with_profit)} profitable regime(s) of "
        f"{len(distinct_regimes)} traded; the largest single-regime share is {dominant_share:.0%}."
    ), breakdown


@dataclass
class WalkForwardFold:
    train_range: tuple
    validation_range: tuple
    test_range: tuple
    train_result: BacktestResult         # fit/in-sample segment
    validation_result: BacktestResult    # first held-out segment (trades gated to this window only)
    out_of_sample_result: BacktestResult # TEST segment -- the final, never-tuned-on holdout
    overfitting_flag: bool               # based on train -> test expectancy collapse
    degradation_ratio: float             # test expectancy / train expectancy
    validation_degradation_ratio: float  # validation expectancy / train expectancy (reported, not gating)
    ml_used: bool

    @property
    def in_sample_result(self) -> BacktestResult:
        """Backward-compatible alias for `train_result` -- 'in-sample' is
        the more general walk-forward term for 'the segment the model/
        strategy was fit on', which is exactly `train_result` now that
        train and validation are reported as separate segments (spec
        section 13) rather than the combined train+validation window this
        used to point to."""
        return self.train_result


@dataclass
class WalkForwardReport:
    folds: List[WalkForwardFold]
    overall_overfitting_detected: bool
    regime_dependency_flag: bool = False
    regime_dependency_note: str = ""
    regime_pnl_breakdown: Dict[str, float] = field(default_factory=dict)   # regime -> total out-of-sample net P&L, across all folds

    def summary(self) -> str:
        lines = [f"Walk-forward validation: {len(self.folds)} fold(s) (TRAIN -> VALIDATION -> TEST, chronological, never shuffled)."]
        for idx, fold in enumerate(self.folds, start=1):
            train = fold.train_result.metrics
            val = fold.validation_result.metrics
            test = fold.out_of_sample_result.metrics
            lines.append(
                f"Fold {idx}: train={fold.train_range}, validation={fold.validation_range}, "
                f"test={fold.test_range}, ML={'on' if fold.ml_used else 'off'}"
            )
            lines.append(
                f"  TRAIN:      PF={train.profit_factor:.2f} exp={train.expectancy:.2f} trades={train.num_trades}"
            )
            lines.append(
                f"  VALIDATION: PF={val.profit_factor:.2f} exp={val.expectancy:.2f} trades={val.num_trades} "
                f"(vs train: {fold.validation_degradation_ratio:.2f}x expectancy)"
            )
            lines.append(
                f"  TEST:       PF={test.profit_factor:.2f} exp={test.expectancy:.2f} trades={test.num_trades} "
                f"(vs train: {fold.degradation_ratio:.2f}x expectancy) | "
                f"{'OVERFITTING SUSPECTED' if fold.overfitting_flag else 'OK'}"
            )
        lines.append(
            "OVERFITTING VERDICT: "
            + ("Overfitting detected in at least one fold's TEST segment -- reject or revise the "
               "strategy before paper trading."
               if self.overall_overfitting_detected else
               "No strong evidence of train->test overfitting across folds. This does NOT guarantee "
               "future performance -- see StrategyHealthReport for a fuller robustness assessment.")
        )
        lines.append(f"REGIME DEPENDENCY (test-window trades only): {self.regime_dependency_note}")
        if self.regime_pnl_breakdown:
            breakdown_str = ", ".join(f"{r}={pnl:.2f}" for r, pnl in sorted(self.regime_pnl_breakdown.items()))
            lines.append(f"  Out-of-sample net P&L by regime: {breakdown_str}")
        return "\n".join(lines)


class WalkForwardValidator:
    def __init__(
        self,
        config: Config,
        backtest_config: Optional[BacktestConfig] = None,
        fundamentals_provider: Optional[HistoricalFundamentalsProvider] = None,
        news_provider: Optional[HistoricalNewsProvider] = None,
        social_provider: Optional[HistoricalSocialProvider] = None,
        ml_enabled: bool = False,
        ml_horizon: int = 5,
        ml_n_splits: int = 5,
        ml_min_auc: float = 0.53,
        ml_label_mode: str = "direction",
        degradation_threshold: float = 0.4,
        min_test_trades: int = 5,
        regime_dominant_fraction_threshold: float = 0.80,
    ):
        self.config = config
        self.backtest_config = backtest_config or BacktestConfig(initial_capital=config.backtesting.initial_capital)
        self.fundamentals_provider = fundamentals_provider
        self.news_provider = news_provider
        self.social_provider = social_provider
        self.ml_enabled = ml_enabled
        self.ml_horizon = ml_horizon
        self.ml_n_splits = ml_n_splits
        self.ml_min_auc = ml_min_auc
        # "direction" (default, unchanged) or "trade_outcome" (spec Part 7 --
        # see backtesting/historical_providers.py's fit_on_training_window).
        self.ml_label_mode = ml_label_mode
        self.degradation_threshold = degradation_threshold
        self.min_test_trades = min_test_trades
        self.regime_dominant_fraction_threshold = regime_dominant_fraction_threshold

    def _make_backtester(self, ml_provider: Optional[HistoricalMLProvider]) -> Backtester:
        return Backtester(
            config=self.config, backtest_config=self.backtest_config,
            fundamentals_provider=self.fundamentals_provider, news_provider=self.news_provider,
            social_provider=self.social_provider, ml_provider=ml_provider,
        )

    @staticmethod
    def _degradation_ratio(baseline_expectancy: float, other_expectancy: float) -> float:
        if baseline_expectancy in (0, None):
            return float("nan")
        return other_expectancy / baseline_expectancy

    def run(
        self,
        symbol: str,
        daily: pd.DataFrame,
        index_daily: Optional[pd.DataFrame] = None,
        macro_daily: Optional[Dict[str, pd.DataFrame]] = None,
        train_bars: int = 500,
        validation_bars: int = 100,
        test_bars: int = 100,
        step_bars: Optional[int] = None,
    ) -> WalkForwardReport:
        daily = daily.sort_index()
        step_bars = step_bars or test_bars
        folds: List[WalkForwardFold] = []
        all_test_trades: List[Trade] = []

        start = 0
        n = len(daily)
        while start + train_bars + validation_bars + test_bars <= n:
            train_df = daily.iloc[start: start + train_bars]
            val_df = daily.iloc[start + train_bars: start + train_bars + validation_bars]
            test_df = daily.iloc[start + train_bars + validation_bars: start + train_bars + validation_bars + test_bars]
            full_through_val = daily.iloc[start: start + train_bars + validation_bars]
            full_through_test = daily.iloc[start: start + train_bars + validation_bars + test_bars]

            ml_provider: Optional[HistoricalMLProvider] = None
            if self.ml_enabled:
                # Fit on TRAIN ONLY -- validation and test must both be genuinely
                # out-of-sample for the model, not just for the strategy's trade
                # filter (spec section 13: report train/validation/test separately).
                ml_provider = HistoricalMLProvider.fit_on_training_window(
                    train_daily=train_df, full_daily=full_through_test,
                    horizon=self.ml_horizon, n_splits=self.ml_n_splits, min_auc=self.ml_min_auc,
                    label_mode=self.ml_label_mode,
                )

            backtester = self._make_backtester(ml_provider)

            # TRAIN: the fitting window itself, taken at face value (this is
            # what the strategy/model saw; not a claim about future performance).
            train_result = backtester.run(symbol, train_df, index_daily=index_daily, macro_daily=macro_daily)

            # VALIDATION: full point-in-time history through the validation
            # window, but only bars inside it may open a position -- the
            # FIRST genuine out-of-sample check.
            validation_result = backtester.run(
                symbol, full_through_val, index_daily=index_daily, macro_daily=macro_daily,
                trade_from=val_df.index[0],
            )

            # TEST: full point-in-time history through the test window, but
            # only bars inside it may open a position -- the FINAL,
            # never-tuned-on holdout.
            out_of_sample_result = backtester.run(
                symbol, full_through_test, index_daily=index_daily, macro_daily=macro_daily,
                trade_from=test_df.index[0],
            )
            all_test_trades.extend(out_of_sample_result.trades)

            train_exp = train_result.metrics.expectancy
            val_exp = validation_result.metrics.expectancy
            test_exp = out_of_sample_result.metrics.expectancy

            overfitting_flag = False
            degradation_ratio = float("nan")
            if out_of_sample_result.metrics.num_trades < self.min_test_trades:
                # Too few test trades to say anything -- explicitly NOT flagged
                # as overfitting (that would be a false positive from thin
                # data, not evidence of an overfit strategy).
                pass
            elif train_exp > 0 and test_exp <= train_exp * self.degradation_threshold:
                overfitting_flag = True
                degradation_ratio = self._degradation_ratio(train_exp, test_exp)
            else:
                degradation_ratio = self._degradation_ratio(train_exp, test_exp)

            validation_degradation_ratio = self._degradation_ratio(train_exp, val_exp)

            folds.append(WalkForwardFold(
                train_range=(train_df.index[0], train_df.index[-1]) if len(train_df) else (None, None),
                validation_range=(val_df.index[0], val_df.index[-1]) if len(val_df) else (None, None),
                test_range=(test_df.index[0], test_df.index[-1]) if len(test_df) else (None, None),
                train_result=train_result,
                validation_result=validation_result,
                out_of_sample_result=out_of_sample_result,
                overfitting_flag=overfitting_flag,
                degradation_ratio=degradation_ratio,
                validation_degradation_ratio=validation_degradation_ratio,
                ml_used=ml_provider is not None,
            ))

            start += step_bars

        overall_flag = any(f.overfitting_flag for f in folds)
        regime_flag, regime_note, regime_breakdown = _regime_dependency_check(
            all_test_trades, dominant_fraction_threshold=self.regime_dominant_fraction_threshold,
        )
        return WalkForwardReport(
            folds=folds, overall_overfitting_detected=overall_flag,
            regime_dependency_flag=regime_flag, regime_dependency_note=regime_note,
            regime_pnl_breakdown=regime_breakdown,
        )
