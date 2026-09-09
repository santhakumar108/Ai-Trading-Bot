"""
AI Signal Engine + Confidence System.

This is the multi-factor decision core described in the spec: it combines
independent component scores (technical, market condition, fundamentals,
news sentiment, social sentiment, volume/price behavior, risk/volatility)
into a single calibrated 0-100 confidence score, and only proposes a
directional trade (BUY/SELL) when:

  * at least `min_independent_signals` of the independent, DIRECTIONAL
    components (technical, fundamentals, news, social, ML) actually have
    real data behind them for this decision,
  * those available components substantially AGREE on direction (low
    "model disagreement"),
  * the weighted confidence -- computed only over AVAILABLE components,
    with weight redistributed away from anything unavailable rather than
    diluted toward a fabricated neutral score -- clears the configured
    threshold, and
  * there is enough history to trust the read at all.

Otherwise the engine returns NO TRADE. NO TRADE is the default outcome of
this function, not an edge case -- callers should expect it most of the
time, by design (see strategy/trade_filter.py for the final gate that can
still veto an engine BUY/SELL for reasons outside this module's view, such
as account-level risk limits).

Availability, not fabrication
------------------------------
`SignalInputs.fundamentals` / `.news` / `.social` / `.ml_probability_up` are
all `Optional`. `None` (or a snapshot whose own quality gate fails -- stale
fundamentals, no credible news, no real mention volume) means "this
component's data was not available for this decision", and it is EXCLUDED
from both the weighted score and the model-agreement vote -- its configured
weight is redistributed proportionally across whatever components ARE
available, rather than being scored as a fabricated 50/neutral value that
would silently count as "no opinion, but still counted." This matters most
in backtesting: point-in-time-correct historical news/social datasets are
rarely available for free, and pretending a symbol had "neutral news" every
day for the last five years would quietly bias every backtested trade
toward the technical/fundamentals view without disclosing it. See
`SignalDecision.unavailable_components` and `.available_independent_signals`
for exactly what was and wasn't used for a given decision.

"Confidence" here is a deterministic, explainable weighted score, not a
statistical probability of profit. Section 6 of the spec requires that this
score be *calibrated* against historical out-of-sample performance before
it is trusted operationally -- see backtesting/walk_forward.py, whose
output (e.g. "trades scored 80-89 historically won X% of the time with
average R of Y") is what turns this number from a heuristic into a
calibrated confidence estimate. Ship this module's raw scores to production
only after that calibration step has been run and reviewed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from config.settings import ConfidenceBands, DecisionThresholds, SignalWeights
from data.macro_data import MacroContext
from fundamentals.fundamental_analysis import FundamentalSnapshot
from indicators.technical import TechnicalSnapshot
from news.news_analysis import NewsAggregate
from sentiment.social_sentiment import SocialAggregate

# Components that count as independent DIRECTIONAL votes for the
# model-agreement check and for the min_independent_signals gate.
# market_condition, volume_price_behavior, and risk_volatility are context
# modifiers (they describe the environment, not "up" vs "down" evidence
# about THIS symbol from an independent source), so they're excluded from
# both -- exactly as before this refactor, just now named explicitly.
DIRECTIONAL_COMPONENTS = ("technical", "fundamentals", "news_sentiment", "social_sentiment", "ml")

# All components that participate in the weighted confidence score.
ALL_COMPONENTS = (
    "technical", "market_condition", "fundamentals", "news_sentiment",
    "social_sentiment", "volume_price_behavior", "risk_volatility",
)


@dataclass
class SignalInputs:
    symbol: str
    technical: TechnicalSnapshot
    market_trend: str                    # "UPTREND" / "DOWNTREND" / "SIDEWAYS" / "UNKNOWN"
    relative_strength: float             # symbol return - index return, e.g. 0.03 = +3%
    fundamentals: Optional[FundamentalSnapshot]
    news: Optional[NewsAggregate]
    social: Optional[SocialAggregate]
    avg_volume_20d: float
    min_liquidity_avg_volume: float
    atr_pct_of_price: float
    max_atr_pct_of_price: float
    min_atr_pct_of_price: float
    history_bars: int
    ml_probability_up: Optional[float] = None   # 0-1, from models/ml_baseline.py, optional
    # "{BULL|BEAR|SIDEWAYS}_{HIGH_VOL|LOW_VOL}" from data/market_data.py's
    # classify_regime(), or "UNKNOWN" with insufficient history. A BEAR
    # regime raises the confidence bar (spec Part 16) -- see
    # SignalEngine.decide()'s effective_min_confidence.
    market_regime: str = "UNKNOWN"
    # Spec Part 2: global + India macro context (US VIX/India VIX/crude/
    # USD-INR trends). None (default) means "not supplied" -- folded into
    # the SAME market_condition score _market_condition_score() already
    # computes, never a separate weighted component (see data/macro_data.py).
    macro_context: Optional[MacroContext] = None


@dataclass
class SignalDecision:
    symbol: str
    component_scores: Dict[str, float]     # each 0-100 -- unavailable components are NOT in this dict
    overall_confidence: float               # 0-100, weighted over available components only
    confidence_label: str
    direction: str                          # "UP", "DOWN", "NONE"
    model_agreement: float                  # 0-1, 1 = perfect agreement among AVAILABLE directional votes
    decision: str                           # "BUY", "SELL", "HOLD", "NO TRADE"
    unavailable_components: List[str] = field(default_factory=list)
    available_independent_signals: int = 0
    reasons: List[str] = field(default_factory=list)
    market_regime: str = "UNKNOWN"

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "component_scores": {k: round(v, 1) for k, v in self.component_scores.items()},
            "overall_confidence": round(self.overall_confidence, 1),
            "confidence_label": self.confidence_label,
            "direction": self.direction,
            "model_agreement": round(self.model_agreement, 2),
            "decision": self.decision,
            "unavailable_components": self.unavailable_components,
            "available_independent_signals": self.available_independent_signals,
            "market_regime": self.market_regime,
            "reasons": self.reasons,
        }


def _market_condition_score(
    market_trend: str, relative_strength: float, macro: Optional[MacroContext] = None,
) -> float:
    """
    "The environment this stock trades in" -- index trend + relative
    strength, PLUS (spec Part 2) global/India macro context when supplied.
    `macro=None` (the default -- every caller that doesn't have macro data
    wired in yet) reproduces the EXACT pre-Phase-6 score, unchanged.

    Macro terms are additive/subtractive, same bounded-points style as the
    rest of this function -- never a separate weighted SignalWeights
    component (see data/macro_data.py's module docstring for why).
    """
    score = 50.0
    if market_trend == "UPTREND":
        score += 20
    elif market_trend == "DOWNTREND":
        score -= 20
    if not np.isnan(relative_strength):
        score += float(np.clip(relative_strength * 200, -20, 20))  # +/-10% RS -> +/-20 pts

    if macro is not None:
        # India VIX: rising = fear rising (headwind); falling = calm (tailwind).
        if macro.india_vix_trend == "UPTREND":
            score -= 10
        elif macro.india_vix_trend == "DOWNTREND":
            score += 5
        # Crude oil: India is a net importer -- rising crude is an import-
        # cost/inflation headwind; falling crude is a tailwind.
        if macro.crude_trend == "UPTREND":
            score -= 8
        elif macro.crude_trend == "DOWNTREND":
            score += 4
        # USD/INR: a rising rate means the rupee is WEAKENING (headwind for
        # import-heavy/foreign-debt names and broad risk sentiment).
        if macro.usdinr_trend == "UPTREND":
            score -= 8
        elif macro.usdinr_trend == "DOWNTREND":
            score += 4
        # US VIX: a global risk-sentiment proxy, once removed from an
        # India-listed stock -- smaller weight than the India-specific terms.
        if macro.us_vix_trend == "UPTREND":
            score -= 5
        elif macro.us_vix_trend == "DOWNTREND":
            score += 3

    return float(np.clip(score, 0, 100))


def _volume_price_behavior_score(avg_volume: float, min_liquidity: float, obv_trend: str, trend: str) -> float:
    score = 50.0
    if avg_volume < min_liquidity:
        score -= 25  # illiquid -> can't trust volume-based confirmation
    else:
        score += 5
    if obv_trend == "RISING" and trend == "UPTREND":
        score += 15
    elif obv_trend == "FALLING" and trend == "DOWNTREND":
        score -= 15  # confirms downside, doesn't help a long
    return float(np.clip(score, 0, 100))


def _confidence_multiplier(
    component: str, fundamentals_ok: bool, fundamentals: Optional[FundamentalSnapshot],
    news_ok: bool, news: Optional[NewsAggregate], social_ok: bool, social: Optional[SocialAggregate],
) -> float:
    """
    Spec Part 15 (decision fusion): within the AVAILABLE components, weight
    each by an EXISTING, already-computed confidence measure rather than
    treating a barely-passing-the-gate reading the same as a fully-
    complete one. Only components with a natural confidence measure are
    reweighted -- technical/market_condition/volume_price_behavior/
    risk_volatility are "always computable directly from real price/volume
    data" (see this module's docstring) and keep multiplier 1.0, same as
    before this existed.

    Clamped to [0.3, 1.0]: the floor means this REDUCES a component's
    influence, it doesn't duplicate the separate exclude/include mechanism
    above (a component that's `_ok` has already cleared its own
    availability gate). The ceiling means it can never AMPLIFY a
    component beyond its configured weight either -- `fundamental_confidence()`
    and `overall_confidence/100` are already self-bounded to [0, 1] by
    their own definitions, but `SocialAggregate.credibility_factor` is a
    caller-supplied value with no such guarantee at its source, so this
    clamp is the one place that promise is actually enforced for every
    component, defensively, regardless of what any individual confidence
    source does.
    """
    if component == "fundamentals" and fundamentals_ok and fundamentals is not None:
        return float(np.clip(fundamentals.fundamental_confidence(), 0.3, 1.0))
    if component == "news_sentiment" and news_ok and news is not None:
        return float(np.clip(float(news.overall_confidence) / 100.0, 0.3, 1.0))
    if component == "social_sentiment" and social_ok and social is not None:
        return float(np.clip(getattr(social, "credibility_factor", 1.0), 0.3, 1.0))
    return 1.0


def _risk_volatility_score(atr_pct: float, max_atr_pct: float, min_atr_pct: float) -> float:
    """Rewards 'normal' volatility, penalizes both dead-quiet (can't define a
    meaningful stop) and abnormally volatile (unpredictable, expensive stops)
    conditions."""
    if atr_pct <= 0 or np.isnan(atr_pct):
        return 30.0
    if atr_pct < min_atr_pct:
        return 35.0
    if atr_pct > max_atr_pct:
        return 15.0
    # Sweet spot roughly in the middle of [min_atr_pct, max_atr_pct]
    mid = (min_atr_pct + max_atr_pct) / 2
    distance = abs(atr_pct - mid) / (max_atr_pct - min_atr_pct + 1e-9)
    return float(np.clip(80 - distance * 40, 40, 80))


def _technical_alpha_score(technical: TechnicalSnapshot) -> float:
    """
    Spec Part 5/6: the "technical" component's score, now genuinely
    derived from the multi-alpha family decomposition
    (`indicators/technical.py`'s `trend_score()`/`momentum_score()`/
    `volatility_volume_score()`, added in an earlier phase but never
    actually consumed by any decision until now) -- equal-weighted, since
    there is no calibration data yet to justify favoring one family over
    another (same "heuristic, not calibrated" honesty already stated in
    this module's docstring).

    Spec Part 6 (correlated-signal control): this stays ONE number feeding
    ONE `component_scores["technical"]` entry and ONE directional lean --
    the three families are blended INTO the single existing "technical"
    vote, never split into three separate votes that would inflate
    `available_independent_signals` or double-count what is fundamentally
    correlated evidence.

    `TechnicalSnapshot.technical_score()` (the older single point-formula)
    is untouched and still used by `backtesting/diagnostics.py`'s
    `SignalFunnel` diagnostic tool -- this function is a SignalEngine-
    internal choice of what to use for decisions, not a replacement of
    that method.
    """
    blend = (technical.trend_score() + technical.momentum_score() + technical.volatility_volume_score()) / 3.0
    return float(np.clip(blend, 0, 100))


class SignalEngine:
    def __init__(self, weights: SignalWeights, thresholds: DecisionThresholds, bands: ConfidenceBands):
        weights.validate()
        self.weights = weights
        self.thresholds = thresholds
        self.bands = bands

    def decide(self, inputs: SignalInputs) -> SignalDecision:
        reasons: List[str] = []
        unavailable: List[str] = []

        # --- Insufficient history is an immediate, unconditional NO TRADE ---
        if inputs.history_bars < self.thresholds.min_history_bars:
            return self._no_trade(
                inputs.symbol,
                [f"Insufficient historical evidence ({inputs.history_bars} bars < "
                 f"{self.thresholds.min_history_bars} required)."],
                market_regime=inputs.market_regime,
            )

        # technical, volume/price, and risk/volatility are always computable
        # directly from price/volume data, so they're never "unavailable".
        # Spec Part 5/6: genuinely derived from the multi-alpha family
        # decomposition now (see _technical_alpha_score's docstring) --
        # still exactly one "technical" vote, never three.
        technical_score = _technical_alpha_score(inputs.technical)

        if inputs.market_trend == "UNKNOWN":
            unavailable.append("market_condition")
            reasons.append("Market/index trend could not be determined; market condition excluded (not faked neutral).")
            market_score = None
        else:
            market_score = _market_condition_score(inputs.market_trend, inputs.relative_strength, inputs.macro_context)

        fundamentals_ok = (
            inputs.fundamentals is not None
            and inputs.fundamentals.data_quality >= 0.4
            and not inputs.fundamentals.is_stale
        )
        if fundamentals_ok:
            fundamental_score = inputs.fundamentals.fundamental_score()
        else:
            unavailable.append("fundamentals")
            reasons.append("Fundamental data unavailable, insufficient, or stale; fundamentals EXCLUDED from scoring (not faked neutral).")
            fundamental_score = None

        news_ok = inputs.news is not None and inputs.news.overall_confidence > 0
        if news_ok:
            news_score = inputs.news.news_score()
            if inputs.news.contradictory:
                reasons.append("Contradictory news reports detected; news signal heavily dampened.")
            if inputs.news.num_credible_sources < 2:
                reasons.append("News signal based on fewer than 2 credible sources; confidence capped.")
        else:
            unavailable.append("news_sentiment")
            reasons.append("No credible/fresh news data available for this decision; news EXCLUDED from scoring (not faked neutral).")
            news_score = None

        social_ok = (
            inputs.social is not None
            and getattr(inputs.social, "data_available", True)
            and inputs.social.mention_volume > 0
        )
        if social_ok:
            social_score = inputs.social.social_score()
            if inputs.social.manipulation_suspected:
                reasons.append("Possible social-media manipulation pattern detected; social signal heavily discounted.")
        else:
            unavailable.append("social_sentiment")
            reasons.append("No social-sentiment data available for this decision; social EXCLUDED from scoring (not faked neutral).")
            social_score = None

        volume_score = _volume_price_behavior_score(
            inputs.avg_volume_20d, inputs.min_liquidity_avg_volume, inputs.technical.obv_trend, inputs.technical.trend
        )
        if inputs.avg_volume_20d < inputs.min_liquidity_avg_volume:
            reasons.append(
                f"Average volume ({inputs.avg_volume_20d:,.0f}) below minimum liquidity threshold "
                f"({inputs.min_liquidity_avg_volume:,.0f})."
            )

        risk_score = _risk_volatility_score(
            inputs.atr_pct_of_price, inputs.max_atr_pct_of_price, inputs.min_atr_pct_of_price
        )
        if inputs.atr_pct_of_price > inputs.max_atr_pct_of_price:
            reasons.append(f"Volatility abnormal (ATR {inputs.atr_pct_of_price:.2%} of price > {inputs.max_atr_pct_of_price:.2%} max).")
        if inputs.atr_pct_of_price < inputs.min_atr_pct_of_price:
            reasons.append(f"Volatility too low to logically define a stop (ATR {inputs.atr_pct_of_price:.2%} of price).")

        raw_scores = {
            "technical": technical_score,
            "market_condition": market_score,
            "fundamentals": fundamental_score,
            "news_sentiment": news_score,
            "social_sentiment": social_score,
            "volume_price_behavior": volume_score,
            "risk_volatility": risk_score,
        }
        # Only available components make it into component_scores / the
        # weighted average -- nothing here is a fabricated stand-in value.
        component_scores = {k: v for k, v in raw_scores.items() if v is not None}

        weight_map = {
            "technical": self.weights.technical,
            "market_condition": self.weights.market_condition,
            "fundamentals": self.weights.fundamentals,
            "news_sentiment": self.weights.news_sentiment,
            "social_sentiment": self.weights.social_sentiment,
            "volume_price_behavior": self.weights.volume_price_behavior,
            "risk_volatility": self.weights.risk_volatility,
        }
        # Spec Part 15 (decision fusion): each AVAILABLE component's
        # configured weight is further scaled by its own confidence, where
        # one exists (fundamentals/news/social -- see _confidence_multiplier).
        # technical/market_condition/volume_price_behavior/risk_volatility
        # keep multiplier 1.0, so this is a pure extension of the existing
        # renormalization below, not a different mechanism.
        confidence_multipliers = {
            k: _confidence_multiplier(k, fundamentals_ok, inputs.fundamentals,
                                       news_ok, inputs.news, social_ok, inputs.social)
            for k in component_scores
        }
        effective_weight = {k: weight_map[k] * confidence_multipliers[k] for k in component_scores}
        available_weight_total = sum(effective_weight[k] for k in component_scores)
        if available_weight_total <= 0:
            # Every scoring component was unavailable -- there is nothing to
            # base a decision on at all.
            reasons.append("No signal components were available for this decision.")
            return self._no_trade(
                inputs.symbol, reasons, unavailable_components=unavailable, market_regime=inputs.market_regime,
            )

        # Renormalize: each available component's (confidence-adjusted)
        # weight is scaled up so the available weights still sum to 1.0,
        # instead of diluting the score toward 50 with a component that was
        # never actually measured.
        overall = sum(component_scores[k] * (effective_weight[k] / available_weight_total) for k in component_scores)
        if unavailable:
            reasons.append(
                f"Weight redistributed across {len(component_scores)} available component(s) "
                f"(excluded: {', '.join(unavailable)})."
            )
        low_confidence = [k for k in component_scores if confidence_multipliers[k] < 0.9]
        if low_confidence:
            reasons.append(
                f"Influence reduced by confidence-weighted fusion for: "
                f"{', '.join(f'{k} (x{confidence_multipliers[k]:.2f})' for k in low_confidence)}."
            )

        # --- Independent-signal count + model / component agreement check ---
        # Convert each *available*, directional component (technical,
        # fundamentals, news, social, and optionally an ML probability) into
        # a -1..+1 lean. Unavailable directional components are simply not
        # counted -- they neither count as agreement nor disagreement.
        directional_leans: List[float] = [(technical_score - 50) / 50]  # technical is always available
        if fundamentals_ok:
            directional_leans.append((fundamental_score - 50) / 50)
        if news_ok:
            directional_leans.append((news_score - 50) / 50)
        if social_ok:
            directional_leans.append((social_score - 50) / 50)
        ml_ok = inputs.ml_probability_up is not None
        if ml_ok:
            directional_leans.append((inputs.ml_probability_up - 0.5) * 2)
        else:
            unavailable.append("ml")

        available_independent_signals = len(directional_leans)

        confidence_label = self.bands.label(overall)

        if available_independent_signals < self.thresholds.min_independent_signals:
            reasons.append(
                f"Only {available_independent_signals} independent signal(s) available "
                f"(technical always counts; fundamentals/news/social/ML count only when real "
                f"data exists) -- need at least {self.thresholds.min_independent_signals} to "
                f"check for agreement. A single available signal is never enough to trade on."
            )
            return SignalDecision(
                symbol=inputs.symbol, component_scores=component_scores, overall_confidence=overall,
                confidence_label=confidence_label, direction="NONE", model_agreement=0.0,
                decision="NO TRADE", unavailable_components=unavailable,
                available_independent_signals=available_independent_signals, reasons=reasons,
                market_regime=inputs.market_regime,
            )

        leans = np.array(directional_leans)
        # Disagreement = normalized standard deviation of the leans (0 = all
        # available votes agree perfectly, higher = more conflict).
        disagreement = float(np.std(leans))
        model_agreement = float(np.clip(1 - disagreement, 0, 1))

        mean_lean = float(np.mean(leans))
        direction = "UP" if mean_lean > 0.05 else ("DOWN" if mean_lean < -0.05 else "NONE")

        if disagreement > self.thresholds.max_model_disagreement:
            reasons.append(
                f"Available signal components disagree significantly (disagreement={disagreement:.2f} > "
                f"{self.thresholds.max_model_disagreement:.2f})."
            )
            return SignalDecision(
                symbol=inputs.symbol, component_scores=component_scores, overall_confidence=overall,
                confidence_label=confidence_label, direction=direction, model_agreement=model_agreement,
                decision="NO TRADE", unavailable_components=unavailable,
                available_independent_signals=available_independent_signals, reasons=reasons,
                market_regime=inputs.market_regime,
            )

        # Spec Part 16: a strongly bearish regime raises the confidence bar
        # -- never lowers it for any other regime. Transparent (logged
        # below when it actually changes the outcome), and computed from
        # the regime alone, never adjusted after seeing whether `overall`
        # would otherwise pass.
        effective_min_confidence = self.thresholds.min_confidence_to_trade
        regime_bumped = inputs.market_regime.startswith("BEAR")
        if regime_bumped:
            effective_min_confidence += self.thresholds.bearish_regime_confidence_bonus

        if overall < effective_min_confidence:
            reasons.append(
                f"Overall confidence {overall:.1f} below minimum required "
                f"{effective_min_confidence:.1f}"
                + (
                    f" (raised from {self.thresholds.min_confidence_to_trade:.1f} by "
                    f"{self.thresholds.bearish_regime_confidence_bonus:.1f} for regime "
                    f"{inputs.market_regime!r})." if regime_bumped else "."
                )
            )
            decision = "NO TRADE" if overall < self.bands.no_trade_max + 1 else "HOLD"
            return SignalDecision(
                symbol=inputs.symbol, component_scores=component_scores, overall_confidence=overall,
                confidence_label=confidence_label, direction=direction, model_agreement=model_agreement,
                decision=decision, unavailable_components=unavailable,
                available_independent_signals=available_independent_signals, reasons=reasons,
                market_regime=inputs.market_regime,
            )

        if direction == "NONE":
            reasons.append("Confidence is high enough but components show no clear directional lean.")
            return SignalDecision(
                symbol=inputs.symbol, component_scores=component_scores, overall_confidence=overall,
                confidence_label=confidence_label, direction=direction, model_agreement=model_agreement,
                decision="HOLD", unavailable_components=unavailable,
                available_independent_signals=available_independent_signals, reasons=reasons,
                market_regime=inputs.market_regime,
            )

        decision = "BUY" if direction == "UP" else "SELL"

        # Spec Part 16: "if technicals say BUY but fresh company news is
        # strongly negative -> NO TRADE or HOLD" (and the symmetric case
        # for SELL). A NARROW, explicit veto -- fires ONLY when news is
        # available+credible (news_ok) AND strongly opposes the direction
        # just reached, checked here (after every other gate already
        # passed) so it can only turn an approval into NO TRADE, never the
        # reverse. This is separate from the multi-source contradiction
        # dampening above (which handles NEWS disagreeing with itself);
        # this handles NEWS disagreeing with the OVERALL decision.
        if news_ok:
            news_conflicts_buy = (
                decision == "BUY" and news_score <= self.thresholds.news_conflict_score_threshold
            )
            news_conflicts_sell = (
                decision == "SELL"
                and news_score >= (100.0 - self.thresholds.news_conflict_score_threshold)
            )
            news_confident_enough = inputs.news.overall_confidence >= self.thresholds.news_conflict_min_confidence
            if (news_conflicts_buy or news_conflicts_sell) and news_confident_enough:
                reasons.append(
                    f"Spec Part 16: credible news (score={news_score:.1f}, "
                    f"confidence={inputs.news.overall_confidence:.1f}) strongly conflicts with the "
                    f"otherwise-approved {decision} -- forced NO TRADE regardless of technical setup."
                )
                return SignalDecision(
                    symbol=inputs.symbol, component_scores=component_scores, overall_confidence=overall,
                    confidence_label=confidence_label, direction=direction, model_agreement=model_agreement,
                    decision="NO TRADE", unavailable_components=unavailable,
                    available_independent_signals=available_independent_signals, reasons=reasons,
                    market_regime=inputs.market_regime,
                )

        reasons.append(f"All gating checks passed: confidence {overall:.1f} ({confidence_label}), "
                        f"model agreement {model_agreement:.2f} across {available_independent_signals} "
                        f"independent signal(s), direction {direction}.")
        return SignalDecision(
            symbol=inputs.symbol, component_scores=component_scores, overall_confidence=overall,
            confidence_label=confidence_label, direction=direction, model_agreement=model_agreement,
            decision=decision, unavailable_components=unavailable,
            available_independent_signals=available_independent_signals, reasons=reasons,
            market_regime=inputs.market_regime,
        )

    def _no_trade(
        self, symbol: str, reasons: List[str], unavailable_components: Optional[List[str]] = None,
        market_regime: str = "UNKNOWN",
    ) -> SignalDecision:
        return SignalDecision(
            symbol=symbol, component_scores={}, overall_confidence=0.0,
            confidence_label="NO TRADE", direction="NONE", model_agreement=0.0,
            decision="NO TRADE", unavailable_components=unavailable_components or [],
            available_independent_signals=0, reasons=reasons, market_regime=market_regime,
        )
