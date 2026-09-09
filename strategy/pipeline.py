"""
Strategy Pipeline -- the SINGLE decision path shared by paper trading and
backtesting.

This module exists to close a specific architectural gap: previously,
`paper_trading/engine.py` called SignalEngine -> TradeRiskCalculator ->
TradeFilter directly, while `backtesting/backtester.py` used a completely
separate, technical-only strategy function with its own inline risk math.
That meant a backtest never actually tested the production decision logic
-- multi-factor scoring, the model-agreement gate, the final approval
checklist -- only a look-alike stand-in for the technical piece of it.

`StrategyPipeline.decide(...)` is now the ONLY place that calls
`SignalEngine.decide`, `TradeRiskCalculator.evaluate`, and
`TradeFilter.approve` in sequence. Both `PaperTradingEngine.scan_symbol`
(paper_trading/engine.py) and `Backtester.run` (backtesting/backtester.py)
build a `strategy.signal_engine.SignalInputs` from whatever data they have
-- live, or point-in-time historical -- and hand it to the SAME
`StrategyPipeline` instance (both constructed via `build_pipeline(config)`
below). Test coverage for this claim lives in
`tests/test_pipeline_parity.py`, which feeds identical inputs through both
call sites and asserts byte-for-byte identical decisions.

What still differs between the two callers, by necessity:
  * WHERE the inputs come from -- live market data vs. a point-in-time
    slice of historical data (never look-ahead; see `backtesting/backtester.py`
    and `backtesting/historical_providers.py`).
  * Account-level state -- `CapitalProtection` -- is owned by each caller
    separately, because a live paper-trading account and a historical
    backtest run are genuinely different accounts with different daily/
    weekly P&L histories. `StrategyPipeline.decide` takes an
    `account_check_fn` callback so this stays true without duplicating the
    account logic itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from config.settings import Config, DecisionThresholds
from data.data_quality import DataQualityReport
from risk.risk_engine import RiskAssessment, TradeRiskCalculator
from strategy.report import TradeReport, build_trade_report
from strategy.signal_engine import ALL_COMPONENTS, SignalDecision, SignalEngine, SignalInputs
from strategy.trade_filter import FilterResult, TradeFilter


@dataclass
class PipelineResult:
    signal: SignalDecision
    risk: Optional[RiskAssessment]
    filter_result: FilterResult
    report: TradeReport


class StrategyPipeline:
    """Bundles the three stages that must never diverge between paper
    trading and backtesting: signal scoring, risk/position sizing, and the
    final approval gate. See module docstring."""

    def __init__(
        self,
        signal_engine: SignalEngine,
        risk_calculator: TradeRiskCalculator,
        trade_filter: TradeFilter,
        decision_thresholds: DecisionThresholds,
    ):
        self.signal_engine = signal_engine
        self.risk_calculator = risk_calculator
        self.trade_filter = trade_filter
        self.decision_thresholds = decision_thresholds

    def propose_stop_target(self, side: str, entry: float, atr: float) -> Tuple[float, float]:
        """
        ATR-based stop/target proposal used when the signal engine approves
        a direction but doesn't itself dictate exact prices. Deliberately
        targets well ABOVE the configured minimum risk/reward (using
        `preferred_risk_reward`, with an extra buffer) rather than exactly
        at the minimum: transaction costs and slippage are subtracted from
        both legs downstream (see TradeRiskCalculator), so a setup sized to
        exactly clear the minimum pre-cost will almost always fall BELOW it
        post-cost and get filtered out for nothing.

        This is the ONE place stop/target proposals are computed -- both
        paper trading and backtesting call this same method, so a change
        here changes both identically.
        """
        stop_multiple = 1.5
        target_multiple = stop_multiple * self.decision_thresholds.preferred_risk_reward * 1.25
        if side == "BUY":
            return entry - stop_multiple * atr, entry + target_multiple * atr
        else:
            return entry + stop_multiple * atr, entry - target_multiple * atr

    def decide(
        self,
        inputs: SignalInputs,
        current_price: float,
        capital: float,
        account_check_fn: Callable[[Optional[RiskAssessment]], List[str]],
        market_trend: str,
        sector_trend: str = "N/A",
        data_quality: Optional[DataQualityReport] = None,
        risk_multiplier: float = 1.0,
    ) -> PipelineResult:
        """
        The one decision path. `current_price` is whatever price the caller
        considers "now" for sizing purposes (a live quote for paper trading;
        the point-in-time bar's close for backtesting) -- neither caller is
        allowed to peek past it. `account_check_fn` receives the computed
        RiskAssessment (or None) and must return a list of account-level
        rejection reasons (daily/weekly loss limits, exposure caps, etc.) --
        empty list means the account-level checks passed.

        `risk_multiplier` (spec Part 26): forwarded as-is to
        `TradeRiskCalculator.evaluate()` -- callers pass their own
        `CapitalProtection.current_risk_multiplier()` here so a losing
        streak shrinks the NEXT trade's size. Defaults to 1.0 (no
        reduction) for callers that don't track a streak.

        `data_quality`, if supplied, is checked FIRST (spec section 22:
        "fail safe" -- market data failure, stale/invalid data, or any other
        data-quality problem must force NO TRADE, never be silently ignored).
        Below `data_quality.is_tradeable()`, every other signal is skipped
        entirely: this is not "one more input to the weighted score," it is
        a hard gate, exactly like the insufficient-history check inside
        SignalEngine itself.
        """
        if data_quality is not None and not data_quality.is_tradeable():
            return self._data_quality_no_trade(inputs, current_price, market_trend, sector_trend, data_quality)

        signal = self.signal_engine.decide(inputs)

        risk_assessment: Optional[RiskAssessment] = None
        if signal.decision in ("BUY", "SELL"):
            stop, target = self.propose_stop_target(signal.decision, current_price, inputs.technical.atr14)
            risk_assessment = self.risk_calculator.evaluate(
                symbol=inputs.symbol, side=signal.decision, entry=current_price,
                stop_loss=stop, target=target, capital=capital,
                confidence=signal.overall_confidence, risk_multiplier=risk_multiplier,
            )

        account_reasons = account_check_fn(risk_assessment)

        news = inputs.news
        social = inputs.social
        filter_result = self.trade_filter.approve(
            signal=signal,
            risk=risk_assessment,
            avg_volume_20d=inputs.avg_volume_20d,
            atr_pct_of_price=inputs.atr_pct_of_price,
            account_level_reasons=account_reasons,
            news_credible_sources=news.num_credible_sources if news else 0,
            news_contradictory=news.contradictory if news else False,
            social_manipulation_suspected=social.manipulation_suspected if social else False,
            news_available=bool(news is not None and len(news.items) > 0),
            market_condition_available="market_condition" in signal.component_scores,
        )

        # Spec Part 4/3 wired into the report (Phase 9): "fundamentals"/
        # "news_sentiment" being present in component_scores IS
        # SignalEngine's own gate for whether that component's data was
        # usable -- reused here rather than re-deriving it.
        fundamental_risk = (
            inputs.fundamentals.fundamental_risk_score()
            if "fundamentals" in signal.component_scores and inputs.fundamentals is not None else None
        )
        news_sentiment_class = (
            inputs.news.sentiment_class
            if "news_sentiment" in signal.component_scores and inputs.news is not None else None
        )

        report = build_trade_report(
            signal=signal, filter_result=filter_result, current_price=current_price,
            market_trend=market_trend, sector_trend=sector_trend, risk=risk_assessment,
            ml_probability_up=inputs.ml_probability_up,
            fundamental_risk=fundamental_risk, news_sentiment_class=news_sentiment_class,
        )
        return PipelineResult(signal=signal, risk=risk_assessment, filter_result=filter_result, report=report)

    def _data_quality_no_trade(
        self, inputs: SignalInputs, current_price: float, market_trend: str, sector_trend: str,
        data_quality: DataQualityReport,
    ) -> PipelineResult:
        issue_summary = "; ".join(f"[{i.severity}] {i.check}: {i.message}" for i in data_quality.issues) or "no specific issues listed"
        reasons = [
            f"Data quality gate failed for {inputs.symbol}: status={data_quality.status}, "
            f"score={data_quality.quality_score:.2f} (min required {data_quality.min_quality_score_to_trade:.2f}). "
            f"{issue_summary}",
            "Forced NO TRADE: this system never trades on data it cannot vouch for, "
            "regardless of what any other signal would have said.",
        ]
        signal = SignalDecision(
            symbol=inputs.symbol, component_scores={}, overall_confidence=0.0,
            confidence_label="NO TRADE", direction="NONE", model_agreement=0.0,
            decision="NO TRADE", unavailable_components=list(ALL_COMPONENTS),
            available_independent_signals=0, reasons=reasons, market_regime=inputs.market_regime,
        )
        filter_result = FilterResult(
            symbol=inputs.symbol, approved=False, final_decision="NO TRADE",
            checklist={"data_quality_acceptable": False}, reasons=reasons,
        )
        report = build_trade_report(
            signal=signal, filter_result=filter_result, current_price=current_price,
            market_trend=market_trend, sector_trend=sector_trend, risk=None,
        )
        return PipelineResult(signal=signal, risk=None, filter_result=filter_result, report=report)


def build_pipeline(config: Config) -> StrategyPipeline:
    """
    The single factory used to construct a StrategyPipeline. Both
    PaperTradingEngine and Backtester call THIS function (never construct
    SignalEngine/TradeRiskCalculator/TradeFilter by hand independently) --
    that's what guarantees they're running identical decision logic, not
    just similarly-configured copies of it.
    """
    signal_engine = SignalEngine(config.signal_weights, config.decision_thresholds, config.confidence_bands)
    risk_calculator = TradeRiskCalculator(
        risk_per_trade_pct=config.risk.risk_per_trade_pct,
        min_risk_reward=config.decision_thresholds.min_risk_reward,
        transaction_cost_pct=config.risk.transaction_cost_pct,
        slippage_pct=config.risk.slippage_pct,
        min_edge_after_costs_pct=config.risk.min_edge_after_costs_pct,
        max_single_position_pct=config.risk.max_single_position_pct,
        default_win_probability=config.risk.default_win_probability,
        min_expected_value_per_share=config.decision_thresholds.min_expected_value_per_share,
        small_account_capital_threshold=config.risk.small_account_capital_threshold,
        small_account_risk_per_trade_pct=config.risk.small_account_risk_per_trade_pct,
    )
    trade_filter = TradeFilter(
        min_risk_reward=config.decision_thresholds.min_risk_reward,
        min_liquidity_avg_volume=config.risk.min_liquidity_avg_volume,
        max_atr_pct_of_price=config.risk.max_atr_pct_of_price,
        min_atr_pct_of_price=config.risk.min_atr_pct_of_price,
        min_confidence_to_trade=config.decision_thresholds.min_confidence_to_trade,
        max_model_disagreement=config.decision_thresholds.max_model_disagreement,
    )
    return StrategyPipeline(signal_engine, risk_calculator, trade_filter, config.decision_thresholds)
