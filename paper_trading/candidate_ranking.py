"""
Account-aware candidate ranking (spec Part 12).

"The system should select based on QUALITY + AFFORDABILITY, not price
alone." `UniverseScanner.scan()` (paper_trading/scanner.py) returns results
in plain symbol-iteration order -- that contract is relied on elsewhere and
is deliberately left untouched. This module is a separate, additive step:
given one scan cycle's results (already computed against the account's
actual capital by `risk/risk_engine.py`'s `TradeRiskCalculator`, which
already knows position_size/affordable_quantity/expected value for that
capital), rank candidates so a cheap, weak, unaffordable-in-practice setup
never outranks a pricier one the account can actually afford with a better
risk-adjusted edge.

Used by `main.py`'s `account-check` and `scan --capital` commands. Does not
duplicate any decision logic -- every number here is read straight off the
`RiskAssessment` that `StrategyPipeline.decide()` already computed.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import List, Optional

from paper_trading.engine import ScanResult


@dataclass
class RankedCandidate:
    symbol: str
    decision: str                          # "BUY" | "SELL" | "HOLD" | "NO TRADE"
    approved: bool
    price: float
    affordable_quantity: int               # spec Part 10-11: max whole shares cash alone affords
    position_size: int                     # final risk-safe, executable quantity
    confidence: float
    expected_value_total: Optional[float]
    risk_reward_ratio: Optional[float]
    rejection_reason: Optional[str]
    # "risk_assessment": affordable_quantity came from TradeRiskCalculator
    # (a direction was reached and costs/risk were actually priced in).
    # "price_only": no direction was ever reached for this symbol this
    # cycle (NO TRADE is this system's default outcome most of the time --
    # see strategy/signal_engine.py), so there is no RiskAssessment to read
    # from; affordable_quantity here is a raw price-only cash estimate
    # (capital // price, ignoring stop distance/costs/risk budget
    # entirely). Distinguishing these two matters: "capital genuinely can't
    # afford this instrument" and "capital could afford it but no qualifying
    # setup existed this cycle" are different findings and must not be
    # collapsed into one misleading "insufficient capital" message.
    affordability_basis: str = "risk_assessment"
    # Spec Part 5.5 (cross-sectional): this symbol's technical_score()
    # percentile-ranked against every OTHER symbol scanned in the SAME
    # cycle (0-100; 100 = strongest technical read of the whole batch).
    # None when this symbol's technical score wasn't available (e.g. the
    # data-quality gate fired before the signal engine ever ran) --
    # informational context only, never a hard gate (see
    # attach_cross_sectional_ranks()).
    technical_score: Optional[float] = None
    technical_percentile: Optional[float] = None
    # Spec Part 7: calibrated P(target before stop) when the live ML
    # trade-outcome model was used (--ml) and produced a usable prediction
    # for this symbol; None otherwise (ML off, insufficient history, or
    # below the AUC usability gate).
    ml_probability: Optional[float] = None


def rank_by_quality_and_affordability(results: List[ScanResult], capital: float) -> List[RankedCandidate]:
    """
    Sorts candidates:
      1. approved (executable) candidates before rejected/NO-TRADE ones,
      2. then by expected value (whole-position, after costs) descending,
      3. then by confidence descending.

    A candidate's own affordable_quantity/position_size/expected_value are
    ALREADY capital-aware where a RiskAssessment exists -- computed by
    TradeRiskCalculator against this account's real balance -- so nothing
    here re-derives or second-guesses that math; it only orders it. Where
    no RiskAssessment exists (no direction was reached this cycle), a raw
    price-only affordability estimate is used instead (see
    RankedCandidate.affordability_basis) so "no signal fired" is never
    reported as "capital insufficient."
    """
    ranked: List[RankedCandidate] = []
    for r in results:
        risk = r.risk
        price = r.report.current_price
        rejection_reason = None
        if not r.filter_result.approved:
            if r.filter_result.reasons:
                rejection_reason = r.filter_result.reasons[0]
            elif r.signal.reasons:
                rejection_reason = r.signal.reasons[0]

        if risk is not None:
            affordable_quantity = risk.affordable_quantity
            position_size = risk.position_size
            expected_value_total = risk.expected_value_total
            risk_reward_ratio = risk.risk_reward_ratio
            affordability_basis = "risk_assessment"
        else:
            affordable_quantity = int(capital // price) if price and price > 0 else 0
            position_size = 0
            expected_value_total = None
            risk_reward_ratio = None
            affordability_basis = "price_only"

        ranked.append(RankedCandidate(
            symbol=r.symbol,
            decision=r.filter_result.final_decision,
            approved=r.filter_result.approved,
            price=price,
            affordable_quantity=affordable_quantity,
            position_size=position_size,
            confidence=r.signal.overall_confidence,
            expected_value_total=expected_value_total,
            risk_reward_ratio=risk_reward_ratio,
            rejection_reason=rejection_reason,
            affordability_basis=affordability_basis,
            technical_score=r.signal.component_scores.get("technical"),
            ml_probability=r.report.ml_probability_up,
        ))

    def sort_key(c: RankedCandidate):
        ev = c.expected_value_total if c.expected_value_total is not None else float("-inf")
        return (not c.approved, -ev, -c.confidence)

    ranked.sort(key=sort_key)
    attach_cross_sectional_ranks(ranked)
    return ranked


def attach_cross_sectional_ranks(ranked: List[RankedCandidate]) -> None:
    """
    Spec Part 5.5: rank each candidate's technical_score against the REST
    OF THE UNIVERSE SCANNED THIS CYCLE (never a fixed benchmark or a single
    hardcoded stock) -- "do not optimize for one stock." Mutates
    `technical_percentile` in place; purely informational context for
    `account-check`'s report, never fed back into the approval gate (a
    weak absolute setup that happens to be the best of a bad batch must
    still be rejected on its own merits -- see strategy/trade_filter.py).
    """
    scored = [c.technical_score for c in ranked if c.technical_score is not None]
    if len(scored) < 2:
        for c in ranked:
            c.technical_percentile = None
        return
    scored_sorted = sorted(scored)
    for c in ranked:
        if c.technical_score is None:
            c.technical_percentile = None
            continue
        # % of the OTHER scanned symbols this candidate's score is >= to.
        rank_position = bisect.bisect_right(scored_sorted, c.technical_score)
        c.technical_percentile = round(100.0 * rank_position / len(scored_sorted), 1)


def affordable_candidates(ranked: List[RankedCandidate]) -> List[RankedCandidate]:
    """Candidates for which at least one whole share is affordable by cash
    alone (spec Part 10-11) -- NOT the same as `approved` (a candidate can
    be affordable but still rejected on quality/risk/EV grounds, and that
    distinction matters for an honest report: 'unaffordable' and 'affordable
    but a bad trade' are different findings)."""
    return [c for c in ranked if c.affordable_quantity >= 1]
