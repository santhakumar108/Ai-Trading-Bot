"""
StrategyHealthReport -- overfitting/robustness detection (spec section 15).

This module does not re-derive statistics research; it combines signals
already computed elsewhere (walk-forward train->test degradation, regime
dependency) with checks it computes itself (trade-count sufficiency,
win-rate plausibility, time-period concentration, single-symbol
concentration, and -- when the caller opts in -- parameter sensitivity)
into a single verdict: HEALTHY / CAUTION / OVERFIT RISK / INSUFFICIENT DATA.

IMPORTANT: a HEALTHY verdict is NOT a claim that the strategy is profitable
or safe to trade with real money -- it only means none of these particular
red flags fired on the data given. A CAUTION or OVERFIT RISK verdict is a
strong signal to not proceed; the absence of one is not permission. Nothing
in this system claims or may claim guaranteed profit (a top-level
constraint of the whole project).

"Excessive tuning" (one of the spec's named checks) is reported as N/A by
design: this codebase does not run automated parameter search/optimization
against historical data, so there is no tuning process here to audit. If a
human manually re-ran a backtest after adjusting thresholds because they
didn't like the first result, THAT is a form of tuning this module cannot
see or police -- the mitigation is procedural (validate on a fresh, later
out-of-sample period after any manual adjustment), not something computable
from a single report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from backtesting.backtester import BacktestResult
from backtesting.walk_forward import WalkForwardReport
from config.settings import Config

VALID_STATUSES = ("PASS", "WARN", "FAIL", "N/A")
VALID_RESULTS = ("HEALTHY", "CAUTION", "OVERFIT RISK", "INSUFFICIENT DATA")


@dataclass
class HealthCheck:
    name: str
    status: str   # one of VALID_STATUSES
    detail: str

    def __post_init__(self):
        if self.status not in VALID_STATUSES:
            raise ValueError(f"HealthCheck status must be one of {VALID_STATUSES}, got {self.status!r}")


@dataclass
class ParameterSensitivityResult:
    unstable: bool
    note: str
    variant_results: Dict[str, float]   # variant label (e.g. '-20%') -> expectancy


@dataclass
class StrategyHealthReport:
    result: str   # one of VALID_RESULTS
    checks: List[HealthCheck]
    summary: str

    def __post_init__(self):
        if self.result not in VALID_RESULTS:
            raise ValueError(f"StrategyHealthReport result must be one of {VALID_RESULTS}, got {self.result!r}")

    def render_text(self) -> str:
        lines = [f"STRATEGY HEALTH: {self.result}", self.summary, ""]
        for c in self.checks:
            lines.append(f"  [{c.status}] {c.name}: {c.detail}")
        return "\n".join(lines)


def _build_summary(result: str, checks: List[HealthCheck]) -> str:
    fails = [c.name for c in checks if c.status == "FAIL"]
    warns = [c.name for c in checks if c.status == "WARN"]
    if result == "OVERFIT RISK":
        return (f"OVERFIT RISK: {len(fails)} check(s) failed ({', '.join(fails)}). "
                "Do not proceed to paper trading (or beyond) on this evidence without revising the "
                "strategy and re-validating on fresh out-of-sample data.")
    if result == "CAUTION":
        return (f"CAUTION: no outright failures, but {len(warns)} check(s) raised a concern "
                f"({', '.join(warns)}). Treat any positive result with real skepticism until these "
                "are understood.")
    return "HEALTHY: no red flags among the checks run. This is NOT a promise of future profit."


def assess_strategy_health(
    walk_forward_report: WalkForwardReport,
    min_test_trades_for_assessment: int = 30,
    max_plausible_win_rate: float = 0.85,
    time_period_dominant_fraction_threshold: float = 0.80,
    per_symbol_net_pnl: Optional[Dict[str, float]] = None,
    symbol_dominant_fraction_threshold: float = 0.80,
    parameter_sensitivity: Optional[ParameterSensitivityResult] = None,
) -> StrategyHealthReport:
    """
    `walk_forward_report` is the ONLY required input -- run
    `WalkForwardValidator.run(...)` first (spec section 13) and pass its
    result here. `per_symbol_net_pnl` (symbol -> total net P&L) and
    `parameter_sensitivity` (from `run_parameter_sensitivity_check` below)
    are optional; omitting them means those two specific checks report N/A
    rather than silently passing.
    """
    checks: List[HealthCheck] = []

    all_test_trades = [t for f in walk_forward_report.folds for t in f.out_of_sample_result.trades]
    total_test_trades = len(all_test_trades)

    # 1. Trade-count sufficiency -- gates everything else if it fails.
    if total_test_trades < min_test_trades_for_assessment:
        checks.append(HealthCheck(
            "trade_count_sufficiency", "FAIL",
            f"Only {total_test_trades} out-of-sample (test-window) trade(s) across "
            f"{len(walk_forward_report.folds)} fold(s) -- fewer than the "
            f"{min_test_trades_for_assessment} needed to draw any statistically meaningful "
            "conclusion. Every other check is informational only until more history/trades exist.",
        ))
        return StrategyHealthReport(
            result="INSUFFICIENT DATA", checks=checks,
            summary=(f"INSUFFICIENT DATA: only {total_test_trades} out-of-sample trade(s) -- collect "
                     "more history (a longer backtest period, or more symbols) before this can be "
                     "assessed one way or the other."),
        )
    checks.append(HealthCheck(
        "trade_count_sufficiency", "PASS",
        f"{total_test_trades} out-of-sample trades across {len(walk_forward_report.folds)} fold(s).",
    ))

    # 2. Train -> test divergence (walk-forward's own overfitting flag).
    if walk_forward_report.overall_overfitting_detected:
        flagged = [i + 1 for i, f in enumerate(walk_forward_report.folds) if f.overfitting_flag]
        checks.append(HealthCheck(
            "train_test_divergence", "FAIL",
            f"Out-of-sample expectancy collapsed relative to train in fold(s) {flagged} -- a classic "
            "overfitting signature.",
        ))
    else:
        checks.append(HealthCheck(
            "train_test_divergence", "PASS",
            "No fold showed a train->test expectancy collapse beyond the configured threshold.",
        ))

    # 3. Win-rate plausibility -- a suspiciously high win rate is a common
    #    symptom of a subtle look-ahead/data leak, not a reason for confidence.
    wins = sum(1 for t in all_test_trades if t.net_pnl > 0)
    win_rate = wins / total_test_trades
    if win_rate >= max_plausible_win_rate:
        checks.append(HealthCheck(
            "win_rate_plausibility", "WARN",
            f"Out-of-sample win rate is {win_rate:.0%}, at or above the {max_plausible_win_rate:.0%} "
            "sanity ceiling -- a real, realistic-cost strategy performing this consistently well is "
            "rare. Treat this as a prompt to re-check for a subtle look-ahead or data leak, not as a "
            "reason for confidence.",
        ))
    else:
        checks.append(HealthCheck(
            "win_rate_plausibility", "PASS",
            f"Out-of-sample win rate ({win_rate:.0%}) is within a plausible range.",
        ))

    # 4. Regime dependency (spec section 14, computed by the walk-forward run itself).
    checks.append(HealthCheck(
        "regime_dependency", "WARN" if walk_forward_report.regime_dependency_flag else "PASS",
        walk_forward_report.regime_dependency_note,
    ))

    # 5. Time-period concentration -- does most of the profit come from one fold?
    fold_pnls = {i: sum(t.net_pnl for t in f.out_of_sample_result.trades)
                 for i, f in enumerate(walk_forward_report.folds, start=1)}
    if len(walk_forward_report.folds) < 2:
        checks.append(HealthCheck(
            "time_period_concentration", "N/A",
            "Only one walk-forward fold was run -- cannot assess whether profitability depends on a "
            "single narrow historical period. Run a longer series (more folds) before trusting this "
            "result across time.",
        ))
    else:
        profitable_folds = {k: v for k, v in fold_pnls.items() if v > 0}
        total_profit = sum(profitable_folds.values())
        if total_profit <= 0:
            checks.append(HealthCheck(
                "time_period_concentration", "WARN",
                "Out-of-sample trading was not net profitable across folds; time-period concentration "
                "is moot until it is.",
            ))
        else:
            dominant_fold, dominant_pnl = max(profitable_folds.items(), key=lambda kv: kv[1])
            share = dominant_pnl / total_profit
            if share >= time_period_dominant_fraction_threshold:
                checks.append(HealthCheck(
                    "time_period_concentration", "WARN",
                    f"{share:.0%} of total out-of-sample profit came from a single fold (fold "
                    f"{dominant_fold}) -- results may depend heavily on one narrow historical period "
                    "rather than a repeatable edge.",
                ))
            else:
                checks.append(HealthCheck(
                    "time_period_concentration", "PASS",
                    f"Profit is spread across folds; the largest single-fold share is {share:.0%}.",
                ))

    # 6. Single-symbol concentration -- only assessable if the caller ran multiple symbols.
    if per_symbol_net_pnl and len(per_symbol_net_pnl) > 1:
        profitable_symbols = {k: v for k, v in per_symbol_net_pnl.items() if v > 0}
        total = sum(profitable_symbols.values())
        if total <= 0:
            checks.append(HealthCheck(
                "symbol_concentration", "WARN",
                "Not net profitable across the symbols assessed; symbol concentration is moot.",
            ))
        else:
            dom_symbol, dom_pnl = max(profitable_symbols.items(), key=lambda kv: kv[1])
            share = dom_pnl / total
            if share >= symbol_dominant_fraction_threshold:
                checks.append(HealthCheck(
                    "symbol_concentration", "WARN",
                    f"{share:.0%} of total profit across {len(per_symbol_net_pnl)} symbols came from "
                    f"{dom_symbol} alone -- do not extrapolate this result to the whole universe.",
                ))
            else:
                checks.append(HealthCheck(
                    "symbol_concentration", "PASS",
                    f"Profit is spread across symbols; the largest single-symbol share is {share:.0%}.",
                ))
    else:
        checks.append(HealthCheck(
            "symbol_concentration", "N/A",
            "Assessed for a single symbol only -- pass `per_symbol_net_pnl` (from backtesting several "
            "symbols) to check for single-stock dependence.",
        ))

    # 7. Parameter sensitivity -- only assessable if the caller opted in.
    if parameter_sensitivity is not None:
        checks.append(HealthCheck(
            "parameter_sensitivity", "FAIL" if parameter_sensitivity.unstable else "PASS",
            parameter_sensitivity.note,
        ))
    else:
        checks.append(HealthCheck(
            "parameter_sensitivity", "N/A",
            "Not assessed here -- for this codebase's actual parameter-sensitivity diagnostic "
            "(spec Part 28), run the `optimize` CLI command (backtesting/sensitivity.py). A caller "
            "can alternatively opt into THIS specific check by calling "
            "`run_parameter_sensitivity_check(...)` directly and passing its result in via "
            "`parameter_sensitivity=`.",
        ))

    # 8. Excessive tuning -- documented limitation, not a computed check (see module docstring).
    checks.append(HealthCheck(
        "excessive_tuning", "N/A",
        "This system does not run automated parameter search/optimization against historical data, "
        "so there is no tuning process to audit here. If thresholds were manually adjusted after "
        "seeing this exact backtest's results, treat that as tuning yourself and re-validate on a "
        "fresh, later out-of-sample period before trusting the outcome.",
    ))

    if any(c.status == "FAIL" for c in checks):
        result = "OVERFIT RISK"
    elif any(c.status == "WARN" for c in checks):
        result = "CAUTION"
    else:
        result = "HEALTHY"

    return StrategyHealthReport(result=result, checks=checks, summary=_build_summary(result, checks))


def run_parameter_sensitivity_check(
    make_config_variant: Callable[[float], Config],
    run_backtest: Callable[[Config], BacktestResult],
    perturbations: Sequence[float] = (-0.20, -0.10, 0.0, 0.10, 0.20),
    instability_threshold: float = 1.5,
) -> ParameterSensitivityResult:
    """
    Generic parameter-sensitivity harness (spec section 15: "unstable
    parameter sensitivity"). `make_config_variant(delta)` must return a
    `Config` with exactly ONE parameter perturbed by the given fractional
    delta (e.g. -0.20 means 20% lower than the baseline value); this
    function is deliberately agnostic about WHICH parameter -- the caller
    decides (a stop-loss ATR multiple, `preferred_risk_reward`,
    `min_confidence_to_trade`, etc.) by how it builds each variant's Config.
    `run_backtest(config)` runs whatever single backtest the caller wants
    (one symbol, one walk-forward fold's test window, a full multi-symbol
    sweep -- this function doesn't care) and must return its `BacktestResult`.

    Flags instability when the SIGN of expectancy flips between variants
    (some profitable, some not, purely from a small parameter nudge), or the
    spread between the best and worst variant's expectancy exceeds
    `instability_threshold` times the baseline (delta=0.0) variant's own
    expectancy magnitude. `0.0` must be included in `perturbations` for the
    spread comparison to have a baseline to compare against.
    """
    if 0.0 not in perturbations:
        raise ValueError("perturbations must include 0.0 as the baseline (unperturbed) variant.")

    variant_results: Dict[str, float] = {}
    for delta in perturbations:
        config = make_config_variant(delta)
        result = run_backtest(config)
        variant_results[f"{delta:+.0%}"] = result.metrics.expectancy

    baseline = variant_results["+0%"]
    signs = {(1 if v > 0 else (-1 if v < 0 else 0)) for v in variant_results.values()}
    sign_flip = len({s for s in signs if s != 0}) > 1

    spread = max(variant_results.values()) - min(variant_results.values())
    baseline_abs = abs(baseline)
    large_spread = baseline_abs > 0 and spread > instability_threshold * baseline_abs

    unstable = sign_flip or large_spread
    if unstable:
        reason = "sign flips between profitable and unprofitable variants" if sign_flip else "spread far exceeds the baseline variant's own expectancy"
        note = (f"Expectancy across perturbed parameter variants is unstable ({reason}): "
                f"{variant_results}. This strategy's edge may be a narrow artifact of the exact "
                "parameter values chosen, not a robust effect -- do not trust a result that only "
                "appears at one precise setting.")
    else:
        note = f"Expectancy is reasonably stable across perturbed parameter variants: {variant_results}."
    return ParameterSensitivityResult(unstable=unstable, note=note, variant_results=variant_results)
