"""
Structured trade-decision report (spec section 13).

Turns a SignalDecision + optional RiskAssessment + FilterResult into the
exact structured report format requested: symbol, price, trend, per-
component scores, entry/stop/target/RR, decision, reasons, and an explicit
invalidation condition. Pure formatting/aggregation -- no new logic.

Component scores are `Optional[float]`: `None` means that component was
UNAVAILABLE for this decision (excluded from scoring and weight-
redistributed, per `strategy.signal_engine`) -- rendered as "N/A", never as
a fabricated 0 or 50.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from risk.risk_engine import RiskAssessment
from strategy.signal_engine import SignalDecision
from strategy.trade_filter import FilterResult


@dataclass
class TradeReport:
    symbol: str
    current_price: float
    market_trend: str
    sector_trend: str
    technical_score: Optional[float]
    fundamental_score: Optional[float]
    news_score: Optional[float]
    social_score: Optional[float]
    risk_score: Optional[float]
    overall_confidence: float
    confidence_label: str
    entry: Optional[float]
    stop_loss: Optional[float]
    target: Optional[float]
    risk_reward: Optional[float]
    expected_risk: Optional[float]
    expected_reward: Optional[float]
    decision: str
    reasons: List[str]
    invalidation_condition: str
    unavailable_components: List[str] = field(default_factory=list)
    market_regime: str = "UNKNOWN"   # spec Part 5.6/16 -- see data/market_data.py's classify_regime()
    # Spec Part 7: calibrated P(target hit before stop), from the ML
    # trade-outcome meta-model -- None when unavailable (ML off, insufficient
    # history, or AUC below the usability gate), never fabricated.
    ml_probability_up: Optional[float] = None
    # Spec Part 4/3, wired into the report (Phase 9): FundamentalSnapshot.
    # fundamental_risk_score() and NewsAggregate.sentiment_class, both None
    # when their component was unavailable/excluded (see fundamental_score/
    # news_score above for the same convention).
    fundamental_risk: Optional[float] = None
    news_sentiment_class: Optional[str] = None
    expected_value_per_share: Optional[float] = None   # spec sections 10 & 18 -- after transaction
                                                        # costs/slippage, at the assumed win
                                                        # probability (see risk.risk_engine); None
                                                        # when there's no risk assessment at all
                                                        # (e.g. HOLD/NO TRADE), never a fabricated 0.
    expected_value_total: Optional[float] = None       # expected_value_per_share * position size

    def render_text(self) -> str:
        def fmt(v, spec=".2f"):
            return format(v, spec) if v is not None else "N/A"

        lines = [
            f"Symbol: {self.symbol}",
            f"Current Price: {fmt(self.current_price)}",
            f"Market Trend: {self.market_trend}",
            f"Sector Trend: {self.sector_trend}",
            f"Market Regime: {self.market_regime}",
            f"Technical Score: {fmt(self.technical_score, '.1f')}/100",
            f"Fundamental Score: {fmt(self.fundamental_score, '.1f')}/100" + (" (unavailable)" if self.fundamental_score is None else ""),
            f"Fundamental Risk: {fmt(self.fundamental_risk, '.1f')}/100" + (" (unavailable)" if self.fundamental_risk is None else ""),
            f"News Score: {fmt(self.news_score, '.1f')}/100" + (" (unavailable)" if self.news_score is None else ""),
            f"News Sentiment: {self.news_sentiment_class or 'N/A'}",
            f"Social Sentiment Score: {fmt(self.social_score, '.1f')}/100" + (" (unavailable)" if self.social_score is None else ""),
            f"Risk Score: {fmt(self.risk_score, '.1f')}/100",
            f"ML Probability (target before stop): {fmt(self.ml_probability_up, '.1%')}"
            + (" (unavailable)" if self.ml_probability_up is None else ""),
            f"Overall Confidence: {fmt(self.overall_confidence, '.1f')}/100 ({self.confidence_label})",
            f"Entry: {fmt(self.entry)}",
            f"Stop Loss: {fmt(self.stop_loss)}",
            f"Target: {fmt(self.target)}",
            f"Risk/Reward: {fmt(self.risk_reward)}",
            f"Expected Risk: {fmt(self.expected_risk)}",
            f"Expected Reward: {fmt(self.expected_reward)}",
            f"Expected Value (per share): {fmt(self.expected_value_per_share, '.4f')}",
            f"Expected Value (position): {fmt(self.expected_value_total)}",
            f"Decision: {self.decision}",
            "Reasons:",
        ]
        for i, r in enumerate(self.reasons[:6], start=1):
            lines.append(f"  {i}. {r}")
        lines.append(f"Invalidation condition: {self.invalidation_condition}")
        if self.unavailable_components:
            lines.append(f"Unavailable components (excluded, not faked): {', '.join(self.unavailable_components)}")
        lines.append(
            "\nDisclaimer: This is a probabilistic, evidence-based estimate, not a "
            "guarantee of profit or a claim of near-certain accuracy."
        )
        return "\n".join(lines)


def _invalidation_condition(signal: SignalDecision, risk: Optional[RiskAssessment]) -> str:
    if risk is not None and risk.stop_loss:
        base = f"Price closes beyond the stop-loss ({risk.stop_loss:.2f}), invalidating the setup."
    else:
        base = "The technical setup that produced this signal no longer holds (e.g. trend reverses)."
    if signal.model_agreement < 0.7:
        base += " Also invalid if remaining signal components diverge further."
    return base


def build_trade_report(
    signal: SignalDecision,
    filter_result: FilterResult,
    current_price: float,
    market_trend: str,
    sector_trend: str = "N/A",
    risk: Optional[RiskAssessment] = None,
    ml_probability_up: Optional[float] = None,
    fundamental_risk: Optional[float] = None,
    news_sentiment_class: Optional[str] = None,
) -> TradeReport:
    expected_risk = risk.capital_at_risk if risk else None
    expected_reward = (
        risk.position_size * risk.reward_per_share if risk and risk.position_size else None
    )
    expected_value_per_share = risk.expected_value_per_share if risk else None
    expected_value_total = risk.expected_value_total if risk else None
    scores = signal.component_scores  # only AVAILABLE components are keys here
    return TradeReport(
        symbol=signal.symbol,
        current_price=current_price,
        market_trend=market_trend,
        sector_trend=sector_trend,
        market_regime=signal.market_regime,
        technical_score=scores.get("technical"),
        fundamental_score=scores.get("fundamentals"),
        news_score=scores.get("news_sentiment"),
        social_score=scores.get("social_sentiment"),
        risk_score=scores.get("risk_volatility"),
        overall_confidence=signal.overall_confidence,
        confidence_label=signal.confidence_label,
        entry=risk.entry if risk else None,
        stop_loss=risk.stop_loss if risk else None,
        target=risk.target if risk else None,
        risk_reward=risk.risk_reward_ratio if risk else None,
        expected_risk=expected_risk,
        expected_reward=expected_reward,
        expected_value_per_share=expected_value_per_share,
        expected_value_total=expected_value_total,
        decision=filter_result.final_decision,
        reasons=filter_result.reasons if not filter_result.approved else signal.reasons,
        invalidation_condition=_invalidation_condition(signal, risk),
        unavailable_components=list(signal.unavailable_components),
        ml_probability_up=ml_probability_up,
        fundamental_risk=fundamental_risk,
        news_sentiment_class=news_sentiment_class,
    )
