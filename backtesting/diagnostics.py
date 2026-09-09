"""
Stage-by-stage decision funnel diagnostics.

This module exists to answer one question precisely: *at which exact stage
do signals disappear for a given symbol/run?* It does not reimplement any
decision logic -- it drives the SAME `strategy.pipeline.StrategyPipeline`
(via `strategy.pipeline.build_pipeline`), the SAME `TechnicalAnalyzer`, and
the SAME `classify_regime` that `backtesting.backtester.Backtester` uses,
and simply tallies, bar by bar, how many bars reach each stage and why bars
that don't reach the next stage were stopped. `Backtester.run()` itself only
reports the FINAL approved-trade count plus a single aggregate
`signals_rejected_at_execution` -- this module fills the gap between "0
approved trades" and "why", per spec section 2's stage list:

    Data received -> Indicator warm-up -> Technical directional signal ->
    Market regime -> Confidence -> Signal alignment -> Trade setup ->
    Risk/reward -> Expected value -> Risk limits -> Final approved trade

Point-in-time discipline is identical to `Backtester.run()`: at bar `i`,
only `daily.iloc[:i+1]` (and the equivalent index/sector slice) is visible.
This module does not execute trades or track equity -- it only counts
decision outcomes -- so it deliberately does not duplicate the fill/exit
simulation in `Backtester.run()`. A symbol whose FINAL approved-trade count
here differs from `Backtester.run()`'s `signals_approved` most likely means
this module is looking at every eligible bar (including bars while a
`Backtester` run would already be holding a position and skip looking for a
new entry) -- see `FunnelReport.note` on `SignalFunnel.run()`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from backtesting.historical_providers import (
    HistoricalFundamentalsProvider,
    HistoricalMLProvider,
    HistoricalNewsProvider,
    HistoricalSocialProvider,
    NoHistoricalFundamentalsProvider,
    NoHistoricalNewsProvider,
    NoHistoricalSocialProvider,
)
from backtesting.backtester import _align, classify_regime
from config.settings import Config
from data.data_quality import DataQualityChecker, DataQualityReport
from data.market_calendar import get_calendar
from data.market_data import MarketDataProvider
from indicators.technical import TechnicalAnalyzer
from strategy.pipeline import StrategyPipeline, build_pipeline
from strategy.signal_engine import SignalInputs, _technical_alpha_score


@dataclass
class FunnelCounts:
    """Bar counts at each stage of spec section 2's pipeline diagram. Every
    count is a count of BARS (not necessarily distinct trades -- a symbol
    flat the whole run can have many bars reach "final_approved" if the
    signal stays valid for a run of consecutive bars; this module does not
    simulate holding a position, see module docstring)."""

    bars_total: int = 0
    bars_with_min_bars_for_indicators: int = 0  # >= 20 bars, TechnicalAnalyzer.analyze() didn't raise
    indicator_warmup_complete: int = 0          # SMA200 non-NaN, i.e. fully warmed up (informational)
    technical_bullish: int = 0
    technical_bearish: int = 0
    technical_neutral: int = 0
    regime_bull: int = 0
    regime_bear: int = 0
    regime_sideways: int = 0
    regime_unknown: int = 0
    no_trade_insufficient_history: int = 0
    no_trade_no_components_available: int = 0
    no_trade_independent_signals_gate: int = 0
    no_trade_disagreement_gate: int = 0
    confidence_pass: int = 0                    # overall_confidence >= min_confidence_to_trade
    confidence_fail: int = 0
    signal_aligned: int = 0                     # SignalEngine returned BUY/SELL (passed every internal gate)
    trade_setup_computed: int = 0                # a RiskAssessment was produced (entry/stop/target/size)
    risk_reward_pass: int = 0
    risk_reward_fail: int = 0
    expected_value_pass: int = 0
    expected_value_fail: int = 0
    account_risk_limits_pass: int = 0
    account_risk_limits_fail: int = 0
    final_approved: int = 0

    def as_dict(self) -> Dict[str, int]:
        return dict(self.__dict__)


@dataclass
class FunnelReport:
    symbol: str
    data_quality: DataQualityReport
    data_quality_tradeable: bool
    counts: FunnelCounts
    example_approved_bar: Optional[dict] = None   # first bar that reached "final_approved", full detail
    rejection_reason_samples: List[str] = field(default_factory=list)  # a few example NO-TRADE reasons, for context

    def summary_row(self) -> Dict[str, object]:
        c = self.counts
        return {
            "symbol": self.symbol,
            "bars": c.bars_total,
            "raw_signals": c.technical_bullish + c.technical_bearish,
            "regime_pass": c.bars_with_min_bars_for_indicators,  # regime is informational-only, never a gate -- see note below
            "technical_pass": c.signal_aligned,
            "risk_pass": c.trade_setup_computed,
            "rr_pass": c.risk_reward_pass,
            "ev_pass": c.expected_value_pass,
            "final_trades": c.final_approved,
        }


class SignalFunnel:
    """Runs one symbol's history through the real production pipeline,
    bar by bar, point-in-time, tallying which stage each bar reaches.
    See module docstring for exactly what this does and does not simulate.
    """

    def __init__(
        self,
        config: Config,
        pipeline: Optional[StrategyPipeline] = None,
        fundamentals_provider: Optional[HistoricalFundamentalsProvider] = None,
        news_provider: Optional[HistoricalNewsProvider] = None,
        social_provider: Optional[HistoricalSocialProvider] = None,
        ml_provider: Optional[HistoricalMLProvider] = None,
        quality_checker: Optional[DataQualityChecker] = None,
    ):
        self.config = config
        self.pipeline = pipeline or build_pipeline(config)
        self.fundamentals_provider = fundamentals_provider or NoHistoricalFundamentalsProvider()
        self.news_provider = news_provider or NoHistoricalNewsProvider()
        self.social_provider = social_provider or NoHistoricalSocialProvider()
        self.ml_provider = ml_provider
        self.quality_checker = quality_checker or DataQualityChecker(
            calendar=get_calendar(config.system.trading_calendar),
            max_missing_row_fraction=config.data_quality.max_missing_row_fraction,
            max_stale_data_days=config.data_quality.max_stale_data_days,
            abnormal_daily_return_threshold=config.data_quality.abnormal_daily_return_threshold,
            min_volume_for_liquidity_check=config.data_quality.min_volume_for_liquidity_check,
            min_quality_score_to_trade=config.data_quality.min_quality_score_to_trade,
        )
        self.technical_analyzer = TechnicalAnalyzer()
        self._data_utils = MarketDataProvider()

    def run(self, symbol: str, daily: pd.DataFrame, index_daily: Optional[pd.DataFrame] = None) -> FunnelReport:
        daily = daily.sort_index()
        counts = FunnelCounts(bars_total=len(daily))

        quality_report = self.quality_checker.check(symbol, daily, as_of=daily.index[-1] if len(daily) else None)
        data_quality_tradeable = quality_report.is_tradeable()

        aligned_index = _align(daily, index_daily)
        example_approved_bar = None
        rejection_samples: List[str] = []

        # If the run-level data-quality gate fails, `Backtester.run()` would
        # simulate NOTHING -- mirror that here so this funnel's counts stay
        # meaningful and comparable (all downstream counts stay 0, not
        # silently computed against data the real pipeline would never see).
        if not data_quality_tradeable:
            return FunnelReport(
                symbol=symbol, data_quality=quality_report, data_quality_tradeable=False, counts=counts,
                rejection_reason_samples=["Run-level data-quality gate failed -- see data_quality issues."],
            )

        for i in range(len(daily)):
            today = daily.index[i]
            history = daily.iloc[: i + 1]
            if len(history) < 20:
                continue

            regime = classify_regime(daily, i)
            if regime.startswith("BULL"):
                counts.regime_bull += 1
            elif regime.startswith("BEAR"):
                counts.regime_bear += 1
            elif regime.startswith("SIDEWAYS"):
                counts.regime_sideways += 1
            else:
                counts.regime_unknown += 1

            try:
                technical = self.technical_analyzer.analyze(symbol, history)
            except ValueError:
                continue
            counts.bars_with_min_bars_for_indicators += 1
            if technical.sma200 is not None:
                counts.indicator_warmup_complete += 1

            # Spec section 11: "don't reimplement decision logic, just
            # tally" -- SignalEngine has bucketed the "technical" component
            # from the trend/momentum/volatility-volume blend since Phase
            # 10, not the older single-formula technical_score() (still
            # used elsewhere, e.g. tests/test_indicators.py, but no longer
            # what actually drives a decision). Reading the same blend here
            # keeps this funnel's own first real stage consistent with
            # what pipeline.decide() below actually computes for this bar.
            tscore = _technical_alpha_score(technical)
            if tscore > 55:
                counts.technical_bullish += 1
            elif tscore < 45:
                counts.technical_bearish += 1
            else:
                counts.technical_neutral += 1

            if aligned_index is not None:
                index_history = aligned_index.iloc[: i + 1]
                market_trend = self._data_utils.market_trend(index_history)
                relative_strength = self._data_utils.relative_strength(history, index_history)
            else:
                market_trend = "UNKNOWN"
                relative_strength = float("nan")

            fundamentals = self.fundamentals_provider.get(symbol, today)
            news = self.news_provider.get(symbol, today)
            social = self.social_provider.get(symbol, today)
            ml_probability_up = self.ml_provider.get(symbol, today) if self.ml_provider else None
            avg_volume = self._data_utils.average_volume(history)

            inputs = SignalInputs(
                symbol=symbol, technical=technical, market_trend=market_trend,
                relative_strength=relative_strength, fundamentals=fundamentals, news=news,
                social=social, avg_volume_20d=avg_volume,
                min_liquidity_avg_volume=self.config.risk.min_liquidity_avg_volume,
                atr_pct_of_price=technical.atr_pct_of_price,
                max_atr_pct_of_price=self.config.risk.max_atr_pct_of_price,
                min_atr_pct_of_price=self.config.risk.min_atr_pct_of_price,
                history_bars=len(history), ml_probability_up=ml_probability_up,
                # Was previously omitted entirely (silently defaulting to
                # "UNKNOWN"), which meant SignalEngine's bearish-regime
                # confidence bump could never fire inside this diagnostic's
                # own pipeline.decide() calls, regardless of the regime
                # `regime` (above) actually classified for this bar --
                # both PaperTradingEngine.scan_symbol() and Backtester.run()
                # correctly thread this through; this diagnostic must too,
                # to genuinely drive the SAME pipeline (see module docstring).
                market_regime=regime,
            )
            current_price = float(history["Close"].iloc[-1])

            def account_check(risk_assessment, _today=today):
                # No live CapitalProtection instance here (this module
                # counts decision-stage outcomes, not account state across a
                # simulated run) -- always "pass" so this stage's count
                # reflects the SIGNAL/RISK pipeline alone. A real backtest
                # run's actual account-level rejections are already
                # reported separately by Backtester.
                return []

            result = self.pipeline.decide(
                inputs, current_price=current_price, capital=self.config.paper_trading.starting_capital,
                account_check_fn=account_check, market_trend=market_trend,
            )
            signal = result.signal

            reason0 = signal.reasons[0] if signal.reasons else ""
            if "Insufficient historical evidence" in reason0:
                counts.no_trade_insufficient_history += 1
                continue
            if any("No signal components were available" in r for r in signal.reasons):
                counts.no_trade_no_components_available += 1
                continue

            # Matches this file's own pattern for every other stage: read
            # the REAL reason text SignalEngine generated rather than
            # recomputing the threshold independently. A static comparison
            # against min_confidence_to_trade would miss the bearish-regime
            # confidence bump SignalEngine applies (signal_engine.py's
            # effective_min_confidence), miscounting a bumped-and-rejected
            # bear-regime bar as a pass.
            if any("below minimum required" in r for r in signal.reasons):
                counts.confidence_fail += 1
            else:
                counts.confidence_pass += 1

            if any("independent signal(s) available" in r for r in signal.reasons):
                counts.no_trade_independent_signals_gate += 1
                if len(rejection_samples) < 5:
                    rejection_samples.append(f"{today.date()}: {reason0}")
                continue
            if any("disagree significantly" in r for r in signal.reasons):
                counts.no_trade_disagreement_gate += 1
                if len(rejection_samples) < 5:
                    rejection_samples.append(f"{today.date()}: {reason0}")
                continue

            if signal.decision in ("BUY", "SELL"):
                counts.signal_aligned += 1
            else:
                if len(rejection_samples) < 5:
                    rejection_samples.append(f"{today.date()}: {reason0}")

            risk = result.risk
            if risk is not None:
                counts.trade_setup_computed += 1
                if risk.risk_reward_ratio >= self.config.decision_thresholds.min_risk_reward:
                    counts.risk_reward_pass += 1
                else:
                    counts.risk_reward_fail += 1
                if risk.expected_value_positive:
                    counts.expected_value_pass += 1
                else:
                    counts.expected_value_fail += 1

            if result.filter_result.checklist.get("account_level_checks_passed", True):
                counts.account_risk_limits_pass += 1
            else:
                counts.account_risk_limits_fail += 1

            if result.filter_result.approved:
                counts.final_approved += 1
                if example_approved_bar is None:
                    example_approved_bar = {
                        "date": str(today.date()),
                        "decision": result.filter_result.final_decision,
                        "entry": risk.entry if risk else None,
                        "stop_loss": risk.stop_loss if risk else None,
                        "target": risk.target if risk else None,
                        "risk_reward_ratio": risk.risk_reward_ratio if risk else None,
                        "expected_value_per_share": risk.expected_value_per_share if risk else None,
                        "position_size": risk.position_size if risk else None,
                        "confidence": signal.overall_confidence,
                        "available_independent_signals": signal.available_independent_signals,
                        "unavailable_components": list(signal.unavailable_components),
                        "reasons": list(signal.reasons),
                    }

        return FunnelReport(
            symbol=symbol, data_quality=quality_report, data_quality_tradeable=True, counts=counts,
            example_approved_bar=example_approved_bar, rejection_reason_samples=rejection_samples,
        )


def render_summary_table(reports: List[FunnelReport]) -> str:
    """Spec section 1's exact table shape:
    Symbol | Bars | Raw Signals | Regime Pass | Technical Pass | Risk Pass | R:R Pass | EV Pass | Final Trades
    """
    headers = ["Symbol", "Bars", "Raw Signals", "Regime Pass*", "Technical Pass", "Risk Pass", "R:R Pass", "EV Pass", "Final Trades"]
    rows = []
    for r in reports:
        row = r.summary_row()
        rows.append([
            row["symbol"], row["bars"], row["raw_signals"], row["regime_pass"], row["technical_pass"],
            row["risk_pass"], row["rr_pass"], row["ev_pass"], row["final_trades"],
        ])
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h)) for i, h in enumerate(headers)]
    lines = [" | ".join(str(h).ljust(w) for h, w in zip(headers, widths))]
    lines.append("-+-".join("-" * w for w in widths))
    for row in rows:
        lines.append(" | ".join(str(v).ljust(w) for v, w in zip(row, widths)))
    lines.append(
        "\n* 'Regime Pass' = bars with >=20 bars of history (enough for TechnicalAnalyzer); "
        "market regime is NEVER a gate in this pipeline (see backtesting/backtester.py's "
        "classify_regime -- it only labels trades for reporting), so it cannot itself cause "
        "zero trades. See each FunnelReport.counts for the full regime_bull/bear/sideways/unknown "
        "breakdown."
    )
    return "\n".join(lines)
