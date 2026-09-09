"""
Parameter sensitivity / overfitting-protection diagnostic (spec Part 28):
"if a parameter performs extremely well only at one value, flag
instability... prefer stable parameter ranges... reject strategies that
depend on a tiny parameter range."

This is explicitly NOT a parameter search or optimizer. Spec Part 29
forbids "selecting parameters using final test results" -- this module
never writes a value back anywhere, never touches config.yaml, and never
even reads TEST-segment results to judge stability (only VALIDATION is
used for that; TEST is computed and shown purely for transparency,
exactly like `backtesting/walk_forward.py`'s own train/validation/test
separation). It sweeps one already-config-driven decision threshold at a
time and reports how much validation-segment performance moves across
that sweep -- a diagnostic for a human to read and decide, not an
automated selection loop.

Reuses `backtesting.walk_forward.WalkForwardValidator` unchanged -- it
already runs point-in-time-safe TRAIN -> VALIDATION -> TEST folds via
`Backtester.run(..., trade_from=...)`. This module is a sweep LOOP around
it (one WalkForwardValidator run per parameter value), not a second
backtest engine.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from backtesting.walk_forward import WalkForwardValidator
from config.settings import Config

logger = logging.getLogger(__name__)

# Sweeps only `decision_thresholds` fields that change WHICH trades fire --
# not risk-sizing knobs like risk.risk_per_trade_pct, which only scales
# position size (already covered by Phase 1's small-account logic) and
# doesn't affect trade selection at all.
DEFAULT_PARAMETER_SWEEPS: Dict[str, List[float]] = {
    "min_confidence_to_trade": [60.0, 70.0, 80.0, 90.0],
    "min_risk_reward": [1.5, 2.0, 3.0, 4.0],
    "max_model_disagreement": [0.20, 0.30, 0.40, 0.50],
}


def _expectancy_and_pf(pnls: Sequence[float]) -> Dict[str, float]:
    arr = np.array(pnls, dtype=float)
    if len(arr) == 0:
        return {"expectancy": 0.0, "profit_factor": 0.0}
    wins = arr[arr > 0]
    losses = arr[arr < 0]
    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())
    pf = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    return {"expectancy": float(arr.mean()), "profit_factor": pf}


@dataclass
class ParameterPoint:
    value: float
    validation_expectancy: float = 0.0
    validation_num_trades: int = 0
    validation_profit_factor: float = 0.0
    # Informational ONLY -- never read by the instability calculation
    # below (spec Part 29: never select/judge using final test results).
    test_expectancy: float = 0.0
    test_num_trades: int = 0
    test_profit_factor: float = 0.0


@dataclass
class ParameterSensitivity:
    parameter: str
    configured_value: float
    points: List[ParameterPoint] = field(default_factory=list)
    coefficient_of_variation: float = float("nan")  # std/|mean| of validation expectancy, traded points only
    unstable: bool = False
    note: str = ""


@dataclass
class SensitivityReport:
    symbol: str
    period: str
    parameters: List[ParameterSensitivity] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Parameter sensitivity diagnostic for {self.symbol} ({self.period}) -- spec Part 28.",
            "This is a DIAGNOSTIC, not a search: nothing here is applied to config automatically. "
            "Stability is judged from VALIDATION segments only; TEST is shown for transparency and "
            "never used to judge stability or pick a value (spec Part 29).",
        ]
        for ps in self.parameters:
            lines.append(f"\n{ps.parameter} (currently configured: {ps.configured_value}):")
            lines.append(f"  {'VALUE':>10}{'VAL_TRADES':>12}{'VAL_EXPECT':>12}{'VAL_PF':>8}"
                          f"{'TEST_TRADES':>13}{'TEST_EXPECT':>13}")
            for p in ps.points:
                lines.append(
                    f"  {p.value:>10}{p.validation_num_trades:>12}{p.validation_expectancy:>12.2f}"
                    f"{p.validation_profit_factor:>8.2f}{p.test_num_trades:>13}{p.test_expectancy:>13.2f}"
                )
            cv_str = f"{ps.coefficient_of_variation:.2f}" if not np.isnan(ps.coefficient_of_variation) else "N/A"
            lines.append(f"  Coefficient of variation (validation expectancy): {cv_str}")
            lines.append(f"  {'UNSTABLE' if ps.unstable else 'stable-looking'}: {ps.note}")
        return "\n".join(lines)


def run_parameter_sensitivity(
    config: Config,
    symbol: str,
    daily: pd.DataFrame,
    index_daily: Optional[pd.DataFrame] = None,
    macro_daily: Optional[Dict[str, pd.DataFrame]] = None,
    sweeps: Optional[Dict[str, List[float]]] = None,
    train_bars: int = 500,
    validation_bars: int = 100,
    test_bars: int = 100,
    unstable_cv_threshold: float = 1.0,
    period: str = "",
) -> SensitivityReport:
    """
    `unstable_cv_threshold` is an explicit, documented HEURISTIC (with only
    4 grid points per parameter there isn't statistical power for a real
    significance test) -- coefficient of variation above this flags
    `unstable=True`. Never treat this as a rigorous statistical claim.
    """
    sweeps = sweeps or DEFAULT_PARAMETER_SWEEPS
    parameters: List[ParameterSensitivity] = []

    for param_name, values in sweeps.items():
        configured_value = getattr(config.decision_thresholds, param_name)
        points: List[ParameterPoint] = []

        for value in values:
            trial_config = copy.deepcopy(config)
            setattr(trial_config.decision_thresholds, param_name, value)

            try:
                validator = WalkForwardValidator(config=trial_config)
                report = validator.run(
                    symbol, daily, index_daily=index_daily, macro_daily=macro_daily,
                    train_bars=train_bars, validation_bars=validation_bars, test_bars=test_bars,
                )
            except Exception as exc:
                logger.warning("Walk-forward failed for %s=%s: %s", param_name, value, exc)
                points.append(ParameterPoint(value=value))
                continue

            val_pnls = [t.net_pnl for f in report.folds for t in f.validation_result.trades]
            test_pnls = [t.net_pnl for f in report.folds for t in f.out_of_sample_result.trades]
            val_stats = _expectancy_and_pf(val_pnls)
            test_stats = _expectancy_and_pf(test_pnls)

            points.append(ParameterPoint(
                value=value,
                validation_expectancy=val_stats["expectancy"], validation_num_trades=len(val_pnls),
                validation_profit_factor=val_stats["profit_factor"],
                test_expectancy=test_stats["expectancy"], test_num_trades=len(test_pnls),
                test_profit_factor=test_stats["profit_factor"],
            ))

        traded_points = [p for p in points if p.validation_num_trades > 0]
        exps = [p.validation_expectancy for p in traded_points]
        if len(exps) >= 2:
            mean_exp = float(np.mean(exps))
            std_exp = float(np.std(exps))
            cv = (std_exp / abs(mean_exp)) if mean_exp != 0 else float("inf")
            unstable = cv > unstable_cv_threshold
            note = (
                f"Validation expectancy varies widely across {param_name}'s tested values "
                f"(CV={cv:.2f} > {unstable_cv_threshold:.2f}) -- this parameter's apparent edge may "
                "depend on a narrow, possibly overfit range rather than a stable one."
                if unstable else
                f"Validation expectancy is relatively consistent across {param_name}'s tested values "
                f"(CV={cv:.2f})."
            )
        elif len(exps) == 1:
            cv = float("nan")
            unstable = False
            note = "Only one tested value produced any validation trades -- not enough points to assess stability."
        else:
            cv = float("nan")
            unstable = False
            note = "No tested value produced any validation trades over this period -- nothing to assess."

        parameters.append(ParameterSensitivity(
            parameter=param_name, configured_value=configured_value, points=points,
            coefficient_of_variation=cv, unstable=unstable, note=note,
        ))

    return SensitivityReport(symbol=symbol, period=period, parameters=parameters)
