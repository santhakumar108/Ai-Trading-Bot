"""
Trade Filter -- the final approval gate (spec section 14).

Every checkbox in the spec maps to an explicit check here. A trade is only
approved if every single one passes; otherwise the final decision is forced
to NO TRADE, regardless of what the signal engine concluded. This is the
one place in the system where "can this actually be traded" is decided --
the signal engine decides "does this look good", the risk engine decides
"how big and at what price", and this module decides "are we actually
allowed to do it right now."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from risk.risk_engine import RiskAssessment
from strategy.signal_engine import SignalDecision


@dataclass
class FilterResult:
    symbol: str
    approved: bool
    final_decision: str   # "BUY", "SELL", "HOLD", "NO TRADE"
    checklist: dict        # check name -> bool
    reasons: List[str] = field(default_factory=list)


class TradeFilter:
    def __init__(
        self,
        min_risk_reward: float,
        min_liquidity_avg_volume: float,
        max_atr_pct_of_price: float,
        min_atr_pct_of_price: float,
        min_confidence_to_trade: float,
        max_model_disagreement: float,
    ):
        self.min_risk_reward = min_risk_reward
        self.min_liquidity_avg_volume = min_liquidity_avg_volume
        self.max_atr_pct_of_price = max_atr_pct_of_price
        self.min_atr_pct_of_price = min_atr_pct_of_price
        self.min_confidence_to_trade = min_confidence_to_trade
        self.max_model_disagreement = max_model_disagreement

    def approve(
        self,
        signal: SignalDecision,
        risk: Optional[RiskAssessment],
        avg_volume_20d: float,
        atr_pct_of_price: float,
        account_level_reasons: List[str],
        news_credible_sources: int,
        news_contradictory: bool,
        social_manipulation_suspected: bool,
        news_available: bool = True,
        market_condition_available: bool = True,
    ) -> FilterResult:
        """
        news_available: whether any news items were fetched/found for this
        symbol at all (regardless of credibility) -- distinguishes "no news
        exists / news wasn't fetched" (not inherently disqualifying) from
        "news exists but none of it is credible or it's contradictory"
        (which IS disqualifying). Defaults to True so callers that don't
        track this explicitly keep the stricter historical behavior.

        market_condition_available: whether a benchmark/index trend could be
        determined for this decision at all (False when
        `SignalInputs.market_trend == "UNKNOWN"` -- no index_symbol
        configured, index data unavailable, or insufficient index history;
        see `strategy/signal_engine.py`, which already excludes
        `market_condition` from `component_scores` in that case rather than
        fabricating a neutral score). Without this flag, an absent
        `component_scores["market_condition"]` would silently read back as
        0 via `.get(..., 0)` and fail `market_conditions_acceptable`
        unconditionally -- turning "we don't have benchmark data" into an
        undisclosed mandatory gate that blocks every trade, exactly the
        failure mode spec section 9 forbids for optional/unavailable
        components. Defaults to True so callers that don't track this
        explicitly keep the stricter historical behavior (same pattern as
        `news_available`). See tests/test_trade_filter.py's
        `test_missing_market_condition_does_not_block_an_otherwise_good_trade`.
        """
        reasons: List[str] = []
        checklist = {}

        checklist["sufficient_liquidity"] = avg_volume_20d >= self.min_liquidity_avg_volume
        checklist["acceptable_volatility"] = (
            self.min_atr_pct_of_price <= atr_pct_of_price <= self.max_atr_pct_of_price
        )
        checklist["strong_technical_setup"] = signal.component_scores.get("technical", 0) >= 60
        if not market_condition_available:
            checklist["market_conditions_acceptable"] = True  # excluded from scoring already -- nothing to be skeptical about
        else:
            checklist["market_conditions_acceptable"] = signal.component_scores.get("market_condition", 0) >= 40
        if not news_available:
            checklist["news_credibility_acceptable"] = True  # nothing to be skeptical about
        else:
            checklist["news_credibility_acceptable"] = (not news_contradictory) and (news_credible_sources >= 1)
        checklist["social_no_manipulation"] = not social_manipulation_suspected
        checklist["ai_signals_aligned"] = signal.model_agreement >= (1 - self.max_model_disagreement)
        checklist["signal_is_directional"] = signal.decision in ("BUY", "SELL")
        checklist["confidence_sufficient"] = signal.overall_confidence >= self.min_confidence_to_trade

        if risk is not None:
            checklist["risk_reward_sufficient"] = risk.risk_reward_ratio >= self.min_risk_reward
            checklist["stop_loss_logical"] = risk.risk_per_share > 0
            checklist["position_size_within_limits"] = risk.position_size > 0
            checklist["positive_edge_after_costs"] = risk.net_expected_reward_per_share > 0
            checklist["expected_value_positive"] = risk.expected_value_positive
        else:
            checklist["risk_reward_sufficient"] = False
            checklist["stop_loss_logical"] = False
            checklist["position_size_within_limits"] = False
            checklist["positive_edge_after_costs"] = False
            checklist["expected_value_positive"] = False
            reasons.append("No risk assessment available (entry/stop/target could not be computed).")

        checklist["account_level_checks_passed"] = len(account_level_reasons) == 0
        if account_level_reasons:
            reasons.extend(account_level_reasons)

        # Ordered so the reasons a person would consider most fundamental
        # (is there even a directional signal / enough confidence / a sane
        # setup) surface before more peripheral checks (spec section 20:
        # "show the strongest rejection reasons"). Any check not listed
        # here (forward-compatible with new checks) is appended after.
        priority = [
            "signal_is_directional", "confidence_sufficient", "ai_signals_aligned",
            "risk_reward_sufficient", "expected_value_positive", "positive_edge_after_costs",
            "stop_loss_logical", "position_size_within_limits",
            "sufficient_liquidity", "acceptable_volatility", "strong_technical_setup",
            "market_conditions_acceptable", "news_credibility_acceptable", "social_no_manipulation",
            "account_level_checks_passed",
        ]
        failed = [name for name, passed in checklist.items() if not passed]
        failed_ordered = [n for n in priority if n in failed] + [n for n in failed if n not in priority]
        for name in failed_ordered:
            reasons.append(f"Failed check: {name.replace('_', ' ')}.")

        approved = len(failed) == 0
        final_decision = signal.decision if approved else "NO TRADE"

        if not approved and not reasons:
            reasons.append("One or more approval-gate checks failed.")

        return FilterResult(
            symbol=signal.symbol,
            approved=approved,
            final_decision=final_decision,
            checklist=checklist,
            reasons=reasons,
        )
