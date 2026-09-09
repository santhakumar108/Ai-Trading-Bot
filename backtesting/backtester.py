"""
Backtesting Engine.

This runs the EXACT SAME production strategy pipeline
(`strategy.pipeline.StrategyPipeline`, built by `strategy.pipeline.build_pipeline`)
that `paper_trading.engine.PaperTradingEngine` uses -- the same SignalEngine,
the same TradeRiskCalculator, the same TradeFilter, configured from the same
`config.settings.Config`. A backtest run and a paper-trading scan of the
same historical date, given the same inputs, produce the identical
SignalDecision / RiskAssessment / FilterResult. See
`tests/test_pipeline_parity.py` for the tests that prove this.

Point-in-time discipline (no look-ahead), enforced structurally:
  * At bar `i`, every input to the pipeline -- technical indicators, market
    trend, fundamentals, news, social sentiment, ML probability -- is
    computed from `daily.iloc[:i+1]` (and the equivalent point-in-time slice
    of the benchmark index) only. Nothing downstream of bar `i` is ever
    touched to make the decision AT bar `i`.
  * A decision made from bar `i`'s close is only ever executed at bar
    `i+1`'s OPEN (with slippage/spread applied), exactly like a real
    end-of-day system would have to. `tests/test_pipeline_parity.py` and
    `tests/test_backtester.py` both assert this directly (decisions don't
    change if future bars are removed from the input frame; fills always
    land on the bar after the signal bar).
  * The `HistoricalMLProvider` (see `backtesting/historical_providers.py`)
    physically refuses to return a prediction for any date at or before its
    training cutoff, independent of what date range it's asked about.

Fundamentals / news / social sentiment during backtests
---------------------------------------------------------
By default, ALL THREE are reported as unavailable for every historical date
(see `backtesting/historical_providers.py`'s `NoHistorical*Provider`
classes) -- NOT fabricated as neutral or positive. `strategy.signal_engine`
excludes unavailable components from both the weighted score and the
model-agreement vote and redistributes their weight, so a default backtest
run is honestly scored on whatever combination of technical / market-
condition / volume / volatility / ML data actually exists for that date,
never on invented sentiment. If you have a genuine point-in-time-correct
fundamentals/news/social dataset, implement the corresponding provider
interface and pass it to `Backtester.__init__` -- do not synthesize one.

Two different "cost" concepts, on purpose
-------------------------------------------
`config.risk.transaction_cost_pct` / `slippage_pct` (part of the main
`Config`, feeding `TradeRiskCalculator` via the shared pipeline) are the
strategy's OWN cost ASSUMPTION, used at decision time to gate "is the edge
still positive after costs" -- exactly as in live/paper trading. `BacktestConfig`
below (`brokerage_pct`, `taxes_pct`, `slippage_pct`, `bid_ask_spread_pct`) is
the REALIZED fill/ledger cost model used to mark actual simulated P&L. In
production you would tune these to match each other; keeping them separate
here mirrors the real distinction between "what the strategy assumes" and
"what actually happens at the exchange."

Regime segmentation and survivorship bias: unchanged from the previous
version of this module -- `classify_regime` is trailing-only (point-in-time
safe), and this module cannot fix a survivorship-biased input universe; that
is the caller's responsibility (use a point-in-time constituent list, not
"today's index members").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as date_type
from typing import Dict, List, Optional

import numpy as np
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
from backtesting.metrics import PerformanceMetrics, compute_metrics
from config.settings import Config
from data.data_quality import DataQualityChecker, DataQualityReport
from data.market_calendar import get_calendar
from data.macro_data import MacroContext
from data.market_data import MarketDataProvider, classify_regime
from indicators.technical import TechnicalAnalyzer
from risk.risk_engine import CapitalProtection
from strategy.pipeline import StrategyPipeline, build_pipeline
from strategy.signal_engine import SignalInputs


@dataclass
class BacktestConfig:
    """Realized fill/ledger cost model -- see module docstring for why this
    is separate from `config.risk.transaction_cost_pct`/`slippage_pct`."""
    brokerage_pct: float = 0.0010
    taxes_pct: float = 0.0005
    slippage_pct: float = 0.0007
    bid_ask_spread_pct: float = 0.0005
    initial_capital: float = 1_000_000.0
    max_holding_days: int = 20


@dataclass
class Trade:
    symbol: str
    side: str
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    stop_loss: float
    target: float
    shares: int
    exit_reason: str   # "TARGET", "STOP", "TIMEOUT"
    gross_pnl: float
    costs: float
    net_pnl: float
    regime_at_entry: str
    confidence_at_entry: float
    unavailable_components: List[str]
    decision_reasons: List[str]


@dataclass
class BenchmarkComparison:
    """Spec section 12: compare the strategy against a NIFTY 50 (or whatever
    `config.universe.index_symbol` resolves to) buy-and-hold, over the SAME
    calendar window. This is a plain return comparison, not a risk-adjusted
    one -- a strategy that is flat/NO-TRADE most of the time is not directly
    comparable to a fully-invested buy-and-hold just because its total
    return is higher; look at it alongside Sharpe/Sortino/Calmar, not
    instead of them."""
    benchmark_available: bool
    symbol: Optional[str] = None
    buy_hold_return_pct: float = 0.0
    buy_hold_cagr_pct: float = 0.0
    strategy_return_pct: float = 0.0
    strategy_cagr_pct: float = 0.0
    outperformance_pct: float = 0.0   # strategy_return_pct - buy_hold_return_pct
    note: str = ""

    def as_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def execution_assumptions(config: Config, backtest_config: "BacktestConfig") -> Dict[str, str]:
    """Spec sections 11 & 24 (point 7): every backtest execution assumption,
    stated in plain language, so a reader doesn't have to reverse-engineer
    them from the simulation loop. Returned fresh each call (not a class
    attribute) so it always reflects the actual config values in force for
    a given run -- attach the result to a report/dashboard rather than
    re-deriving it by hand."""
    return {
        "fill_timing": (
            "A signal approved on bar i's CLOSE is filled at bar i+1's OPEN, never at bar i's own "
            "close and never at an exact historical close price chosen with hindsight. If bar i is "
            "the last bar in the series, no trade is opened (there is no i+1 to fill at)."
        ),
        "gaps": (
            "The fill price is bar i+1's actual Open, which can gap through the previously computed "
            "stop/target levels. The stop/target themselves are NOT re-priced off the gapped open -- "
            "they stay at the absolute levels approved off bar i's close -- but the fill is re-checked "
            "against the shared risk calculator using the REAL fill price; if the gap already moved "
            "price enough that the trade no longer clears the minimum risk/reward, the trade is "
            "skipped (see `signals_rejected_at_execution`), not forced through anyway."
        ),
        "exit_priority": (
            "On any bar with an open position, a stop-loss or target is detected from that bar's "
            "High/Low only (never intra-bar sequencing, which historical daily OHLCV cannot tell you) "
            "-- a BUY exits on STOP if the bar's Low touched it, else on TARGET if the bar's High "
            "touched it; if both could technically have been touched intra-bar, this simulation "
            "resolves the stop first, which is the conservative assumption for a long position."
        ),
        "timeout_exit": (
            f"A position still open after {backtest_config.max_holding_days} bars is closed at that "
            "bar's Close with exit_reason='TIMEOUT', regardless of stop/target."
        ),
        "slippage_and_spread": (
            f"Every fill (entry and exit) is adjusted by "
            f"{backtest_config.slippage_pct:.4%} slippage plus half the "
            f"{backtest_config.bid_ask_spread_pct:.4%} bid-ask spread, applied against the position "
            "(worse for the trade, never better) -- this is a cost model, not a prediction of the "
            "exact fill a real order would have received."
        ),
        "transaction_costs": (
            f"Brokerage ({backtest_config.brokerage_pct:.4%}) and taxes/STT ({backtest_config.taxes_pct:.4%}) "
            "are charged on BOTH the entry and exit notional, deducted from gross P&L to get net P&L. "
            "This is separate from `config.risk.transaction_cost_pct`/`slippage_pct`, which is the "
            "strategy's OWN cost ASSUMPTION used at decision time to gate 'is the edge still positive "
            "after costs' -- see the module docstring's 'Two different cost concepts' note."
        ),
        "partial_fills": (
            "NOT modeled. Every approved trade fills for its full computed position size at the single "
            "next-bar-open price -- there is no partial-fill or liquidity-exhaustion simulation. This "
            "is a known simplification (see spec section 11's 'partial fills where practical'); for a "
            "small-cap/low-liquidity NSE stock, a real order could receive a materially worse average "
            "fill than this backtest assumes."
        ),
        "position_sizing": (
            "Position size is computed by the SAME `TradeRiskCalculator.evaluate()` used by paper "
            "trading, from `config.risk.risk_per_trade_pct` of current capital and the stop distance -- "
            "not a fixed share count and not 'all-in'."
        ),
        "daily_and_weekly_loss_limits": (
            "The same `CapitalProtection` used by paper trading enforces "
            "`config.risk.max_daily_loss_pct` / `max_weekly_loss_pct`; once breached on a simulated "
            "day/week, no new positions open until the next day/week, exactly as in paper trading."
        ),
        "concurrent_positions_and_sector_exposure": (
            f"At most one open position PER SYMBOL is simulated by this single-symbol `run()` call "
            f"(a multi-symbol portfolio backtest must call `run()` once per symbol and combine the "
            f"results itself); `config.risk.max_simultaneous_positions` and "
            f"`config.risk.max_sector_exposure_pct` are enforced by `CapitalProtection` exactly as in "
            f"paper trading for whatever symbols/sectors are passed to it, but this module does not "
            f"itself orchestrate a multi-symbol run."
        ),
        "trading_calendar_and_market_hours": (
            f"Bars are iterated in the order given -- this module trusts `daily`'s own index rather "
            f"than re-deriving trading days, but the run-level `DataQualityChecker` (built from "
            f"`get_calendar(config.system.trading_calendar)`, currently {config.system.trading_calendar!r}) "
            f"flags missing-row and holiday-mismatch issues against that calendar before any bar is "
            f"simulated. Intraday market-hours (09:15-15:30 IST for NSE) are not applicable here: this "
            f"is a daily-bar backtester, not an intraday one."
        ),
        "data_quality_gate": (
            "A SINGLE run-level data-quality check runs over the WHOLE input series before any bar is "
            "simulated (see `BacktestResult.is_valid`/`data_quality`) -- this is coarser than a "
            "per-bar/point-in-time-recomputed quality gate; a dataset bad enough to fail this check "
            "is not simulated bar-by-bar at all, but a dataset that PASSES the run-level check is not "
            "re-checked bar-by-bar for, e.g., a single corrupted bar in the middle of an otherwise "
            "clean series (a known, documented simplification)."
        ),
        "corporate_actions": (
            "This module does not itself adjust for splits/bonuses/dividends -- it trusts whatever "
            "`daily` price series it is given (adjusted or unadjusted is the caller's choice via "
            "`MarketDataProvider`). An unadjusted series with an unexplained corporate action inside "
            "the backtest window can look like an abnormal price jump to the data-quality checker."
        ),
    }


@dataclass
class BacktestResult:
    trades: List[Trade]
    equity_curve: List[float]
    dates: List[pd.Timestamp]
    metrics: PerformanceMetrics
    metrics_by_regime: Dict[str, PerformanceMetrics]
    signals_approved: int
    signals_rejected_at_execution: int   # approved by the pipeline, but the actual next-bar
                                          # fill no longer cleared risk/reward -- a real,
                                          # legitimate outcome, not a bug
    data_quality: Optional[DataQualityReport] = None   # always populated -- spec section 3:
                                                        # "every backtest run must display data
                                                        # quality status"
    is_valid: bool = True   # False means data_quality failed the run-level gate and NOTHING
                             # below was actually simulated -- trades/metrics are empty, not
                             # "the strategy took no trades"
    benchmark_comparison: Optional[BenchmarkComparison] = None   # spec section 12: NIFTY 50 (or
                                                                  # configured index) buy-and-hold,
                                                                  # same calendar window
    assumptions: Dict[str, str] = field(default_factory=dict)    # spec sections 11 & 24(7): every
                                                                  # execution assumption, in plain
                                                                  # language -- see
                                                                  # `execution_assumptions()`


def _align(daily: pd.DataFrame, other: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Reindexes `other` (e.g. a benchmark index or sector index series) onto
    `daily`'s date index, forward-filling gaps (different exchange holidays,
    etc.). Forward-fill only ever uses PAST values, so this cannot leak
    future information."""
    if other is None:
        return None
    return other.reindex(daily.index, method="ffill")


class Backtester:
    def __init__(
        self,
        config: Config,
        backtest_config: Optional[BacktestConfig] = None,
        fundamentals_provider: Optional[HistoricalFundamentalsProvider] = None,
        news_provider: Optional[HistoricalNewsProvider] = None,
        social_provider: Optional[HistoricalSocialProvider] = None,
        ml_provider: Optional[HistoricalMLProvider] = None,
        pipeline: Optional[StrategyPipeline] = None,
        quality_checker: Optional[DataQualityChecker] = None,
    ):
        self.config = config
        self.backtest_config = backtest_config or BacktestConfig(initial_capital=config.backtesting.initial_capital)
        self.quality_checker = quality_checker or DataQualityChecker(
            calendar=get_calendar(config.system.trading_calendar),
            max_missing_row_fraction=config.data_quality.max_missing_row_fraction,
            max_stale_data_days=config.data_quality.max_stale_data_days,
            abnormal_daily_return_threshold=config.data_quality.abnormal_daily_return_threshold,
            min_volume_for_liquidity_check=config.data_quality.min_volume_for_liquidity_check,
            min_quality_score_to_trade=config.data_quality.min_quality_score_to_trade,
        )
        # THE key line for architectural parity with paper trading: this is
        # the same build_pipeline() factory PaperTradingEngine uses, from
        # the same Config. Passing `pipeline` explicitly (tests do this) lets
        # a caller prove the literal same StrategyPipeline instance is used
        # by both a Backtester and a PaperTradingEngine in the same test.
        self.pipeline = pipeline or build_pipeline(config)
        self.fundamentals_provider = fundamentals_provider or NoHistoricalFundamentalsProvider()
        self.news_provider = news_provider or NoHistoricalNewsProvider()
        self.social_provider = social_provider or NoHistoricalSocialProvider()
        self.ml_provider = ml_provider  # None is a valid, explicit "no ML component"
        self.technical_analyzer = TechnicalAnalyzer()
        self._data_utils = MarketDataProvider()  # used only for its pure, stateless helper methods

    def run(
        self,
        symbol: str,
        daily: pd.DataFrame,
        index_daily: Optional[pd.DataFrame] = None,
        sector: Optional[str] = None,
        sector_daily: Optional[pd.DataFrame] = None,
        trade_from: Optional[pd.Timestamp] = None,
        index_symbol: Optional[str] = None,
        macro_daily: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> BacktestResult:
        """
        `daily` is iterated bar by bar; at each bar `i`, only
        `daily.iloc[:i+1]` (and the equivalent slice of `index_daily` /
        `sector_daily`) is visible to the pipeline. `trade_from`, if given,
        restricts which bars are allowed to OPEN a new position (used by
        `WalkForwardValidator` to give the pipeline a full point-in-time
        history for context while only counting trades in the held-out test
        window) -- it does not hide any data, it only gates execution.

        Before any of that: a RUN-LEVEL data-quality check (spec section 3)
        runs once over the whole `daily` series. Below the configured
        threshold, the run is marked invalid and NOTHING is simulated --
        this is intentionally coarser than a per-bar/point-in-time quality
        gate (see the module docstring's "Known limitations" note); a
        dataset that's corrupted badly enough to fail this check isn't one
        worth backtesting bar-by-bar regardless.

        `index_symbol` is used ONLY for display in `BacktestResult.benchmark_comparison`
        (defaults to `config.universe.index_symbol`, e.g. '^NSEI') -- the actual
        buy-and-hold numbers come from `index_daily` itself, whatever symbol it is.
        The returned `BacktestResult.assumptions` (see `execution_assumptions()`)
        documents every fill/cost/risk/calendar assumption this simulation makes,
        in plain language, per spec sections 11 and 24.
        """
        daily = daily.sort_index()
        index_symbol = index_symbol or self.config.universe.index_symbol
        assumptions = execution_assumptions(self.config, self.backtest_config)

        quality_report = self.quality_checker.check(symbol, daily, as_of=daily.index[-1] if len(daily) else None)
        if not quality_report.is_tradeable():
            return BacktestResult(
                trades=[], equity_curve=[], dates=[],
                metrics=compute_metrics([], [], [], self.backtest_config.initial_capital, 0),
                metrics_by_regime={}, signals_approved=0, signals_rejected_at_execution=0,
                data_quality=quality_report, is_valid=False,
                benchmark_comparison=BenchmarkComparison(
                    benchmark_available=False,
                    note="Run invalidated by the data-quality gate before any simulation or "
                         "benchmark comparison was attempted.",
                ),
                assumptions=assumptions,
            )

        aligned_index = _align(daily, index_daily)
        aligned_sector = _align(daily, sector_daily)
        # Spec Part 2: global + India macro context. Unlike fundamentals/
        # news/social (no free point-in-time dataset), macro series ARE
        # just OHLCV-like data -- aligned the SAME way index_daily/
        # sector_daily already are, via the same `_align()` helper.
        # `macro_daily=None` (every existing caller) means this dict is
        # empty and macro_context is always None below, byte-identical to
        # pre-Phase-6 behavior.
        aligned_macro: Dict[str, pd.DataFrame] = {
            key: series for key, series in (
                (k, _align(daily, v)) for k, v in (macro_daily or {}).items()
            ) if series is not None
        }
        trade_from_ts = pd.Timestamp(trade_from) if trade_from is not None else None

        capital = self.backtest_config.initial_capital
        capital_protection = CapitalProtection(
            starting_capital=capital,
            max_daily_loss_pct=self.config.risk.max_daily_loss_pct,
            max_weekly_loss_pct=self.config.risk.max_weekly_loss_pct,
            max_simultaneous_positions=self.config.risk.max_simultaneous_positions,
            max_sector_exposure_pct=self.config.risk.max_sector_exposure_pct,
            max_single_position_pct=self.config.risk.max_single_position_pct,
            today=daily.index[0].date(),
            max_consecutive_losses_before_reduction=self.config.risk.max_consecutive_losses_before_reduction,
            risk_reduction_factor=self.config.risk.risk_reduction_factor,
        )

        equity_curve: List[float] = []
        dates: List[pd.Timestamp] = []
        trades: List[Trade] = []
        period_returns: List[float] = []

        open_trade = None
        prev_equity = capital
        signals_approved = 0
        signals_rejected_at_execution = 0

        for i in range(len(daily)):
            today = daily.index[i]
            history = daily.iloc[: i + 1]
            regime = classify_regime(daily, i)

            # --- Manage an open position first (point-in-time: only this bar's H/L) ---
            if open_trade is not None:
                bar = daily.iloc[i]
                exit_price = None
                exit_reason = None

                if open_trade["side"] == "BUY":
                    if bar["Low"] <= open_trade["stop_loss"]:
                        exit_price = open_trade["stop_loss"]
                        exit_reason = "STOP"
                    elif bar["High"] >= open_trade["target"]:
                        exit_price = open_trade["target"]
                        exit_reason = "TARGET"
                else:
                    if bar["High"] >= open_trade["stop_loss"]:
                        exit_price = open_trade["stop_loss"]
                        exit_reason = "STOP"
                    elif bar["Low"] <= open_trade["target"]:
                        exit_price = open_trade["target"]
                        exit_reason = "TARGET"

                holding_days = i - open_trade["entry_index"]
                if exit_price is None and holding_days >= self.backtest_config.max_holding_days:
                    exit_price = float(bar["Close"])
                    exit_reason = "TIMEOUT"

                if exit_price is not None:
                    slip = exit_price * (self.backtest_config.slippage_pct + self.backtest_config.bid_ask_spread_pct / 2)
                    exit_price_after_slip = exit_price - slip if open_trade["side"] == "BUY" else exit_price + slip
                    shares = open_trade["shares"]
                    if open_trade["side"] == "BUY":
                        gross_pnl = (exit_price_after_slip - open_trade["entry_price"]) * shares
                    else:
                        gross_pnl = (open_trade["entry_price"] - exit_price_after_slip) * shares

                    cost_rate = self.backtest_config.brokerage_pct + self.backtest_config.taxes_pct
                    costs = (open_trade["entry_price"] + exit_price_after_slip) * shares * cost_rate
                    net_pnl = gross_pnl - costs
                    capital += net_pnl

                    trades.append(Trade(
                        symbol=symbol, side=open_trade["side"], entry_date=open_trade["entry_date"],
                        entry_price=open_trade["entry_price"], exit_date=today, exit_price=exit_price_after_slip,
                        stop_loss=open_trade["stop_loss"], target=open_trade["target"], shares=shares,
                        exit_reason=exit_reason, gross_pnl=gross_pnl, costs=costs, net_pnl=net_pnl,
                        regime_at_entry=open_trade["regime"], confidence_at_entry=open_trade["confidence"],
                        unavailable_components=open_trade["unavailable_components"],
                        decision_reasons=open_trade["decision_reasons"],
                    ))
                    capital_protection.register_close(
                        sector, open_trade["entry_price"] * shares, net_pnl, today=today.date(),
                    )
                    open_trade = None

            # --- Look for a new entry only if flat, and only from `trade_from` onward ---
            can_open_here = open_trade is None and i + 1 < len(daily) and (
                trade_from_ts is None or today >= trade_from_ts
            )
            if can_open_here and len(history) >= 20:
                try:
                    technical = self.technical_analyzer.analyze(symbol, history)
                except ValueError:
                    technical = None

                if technical is not None:
                    if aligned_index is not None:
                        index_history = aligned_index.iloc[: i + 1]
                        market_trend = self._data_utils.market_trend(index_history)
                        relative_strength = self._data_utils.relative_strength(history, index_history)
                    else:
                        market_trend = "UNKNOWN"
                        relative_strength = float("nan")

                    sector_trend = "N/A"
                    if aligned_sector is not None:
                        sector_history = aligned_sector.iloc[: i + 1]
                        sector_trend = self._data_utils.market_trend(sector_history)

                    fundamentals = self.fundamentals_provider.get(symbol, today)
                    news = self.news_provider.get(symbol, today)
                    social = self.social_provider.get(symbol, today)
                    ml_probability_up = self.ml_provider.get(symbol, today) if self.ml_provider else None

                    macro_context = None
                    if aligned_macro:
                        macro_context = MacroContext(
                            us_vix_trend=self._data_utils.market_trend(aligned_macro["us_vix"].iloc[: i + 1])
                            if "us_vix" in aligned_macro else "UNKNOWN",
                            india_vix_trend=self._data_utils.market_trend(aligned_macro["india_vix"].iloc[: i + 1])
                            if "india_vix" in aligned_macro else "UNKNOWN",
                            crude_trend=self._data_utils.market_trend(aligned_macro["crude"].iloc[: i + 1])
                            if "crude" in aligned_macro else "UNKNOWN",
                            usdinr_trend=self._data_utils.market_trend(aligned_macro["usdinr"].iloc[: i + 1])
                            if "usdinr" in aligned_macro else "UNKNOWN",
                        )

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
                        market_regime=regime, macro_context=macro_context,
                    )

                    current_price = float(history["Close"].iloc[-1])

                    def account_check(risk_assessment, _sector=sector, _today=today):
                        notional = risk_assessment.notional_exposure if risk_assessment else 0.0
                        return capital_protection.pre_trade_check(
                            symbol=symbol, sector=_sector, notional_exposure=notional, today=_today.date(),
                        )

                    result = self.pipeline.decide(
                        inputs, current_price=current_price, capital=capital,
                        account_check_fn=account_check, market_trend=market_trend, sector_trend=sector_trend,
                        risk_multiplier=capital_protection.current_risk_multiplier(),
                    )

                    if result.filter_result.approved and result.filter_result.final_decision in ("BUY", "SELL"):
                        signals_approved += 1
                        side = result.filter_result.final_decision
                        proposed_risk = result.risk  # entry/stop/target computed at bar i's close

                        next_bar = daily.iloc[i + 1]
                        raw_entry = float(next_bar["Open"])
                        slip = raw_entry * (self.backtest_config.slippage_pct + self.backtest_config.bid_ask_spread_pct / 2)
                        entry_price = raw_entry + slip if side == "BUY" else raw_entry - slip

                        # Re-price with the SAME risk_calculator (shared with
                        # paper trading), the SAME absolute stop/target
                        # levels that were approved, but the ACTUAL fill
                        # price -- a real overnight gap can move the fill
                        # enough that the trade no longer clears the
                        # risk/reward bar, exactly as it would for a live
                        # order the next morning.
                        executed_risk = self.pipeline.risk_calculator.evaluate(
                            symbol=symbol, side=side, entry=entry_price,
                            stop_loss=proposed_risk.stop_loss, target=proposed_risk.target, capital=capital,
                            risk_multiplier=capital_protection.current_risk_multiplier(),
                        )

                        if executed_risk.approved:
                            open_trade = {
                                "side": side, "entry_date": daily.index[i + 1], "entry_price": entry_price,
                                "stop_loss": proposed_risk.stop_loss, "target": proposed_risk.target,
                                "shares": executed_risk.position_size, "entry_index": i + 1, "regime": regime,
                                "confidence": result.signal.overall_confidence,
                                "unavailable_components": list(result.signal.unavailable_components),
                                "decision_reasons": list(result.signal.reasons),
                            }
                            capital_protection.register_open(sector, executed_risk.notional_exposure)
                        else:
                            signals_rejected_at_execution += 1

            equity_curve.append(capital)
            dates.append(today)
            if prev_equity != 0:
                period_returns.append((capital - prev_equity) / prev_equity)
            else:
                period_returns.append(0.0)
            prev_equity = capital

        trade_pnls = [t.net_pnl for t in trades]
        overall_metrics = compute_metrics(
            trade_pnls, equity_curve, period_returns, self.backtest_config.initial_capital, len(daily),
            dates=dates,
        )

        metrics_by_regime: Dict[str, PerformanceMetrics] = {}
        regimes = sorted({t.regime_at_entry for t in trades})
        for regime in regimes:
            regime_pnls = [t.net_pnl for t in trades if t.regime_at_entry == regime]
            if regime_pnls:
                # No `dates=` here on purpose -- this is a synthetic cumulative-P&L series with
                # gaps between entries, not a real daily equity curve, so monthly/yearly return
                # buckets from it would be meaningless (see compute_metrics' docstring).
                metrics_by_regime[regime] = compute_metrics(
                    regime_pnls, list(np.cumsum(regime_pnls) + self.backtest_config.initial_capital),
                    [p / self.backtest_config.initial_capital for p in regime_pnls],
                    self.backtest_config.initial_capital, len(regime_pnls),
                )

        benchmark_comparison = self._compute_benchmark_comparison(
            aligned_index, dates, overall_metrics, index_symbol,
        )

        return BacktestResult(
            trades=trades, equity_curve=equity_curve, dates=dates,
            metrics=overall_metrics, metrics_by_regime=metrics_by_regime,
            signals_approved=signals_approved, signals_rejected_at_execution=signals_rejected_at_execution,
            data_quality=quality_report, is_valid=True,
            benchmark_comparison=benchmark_comparison, assumptions=assumptions,
        )

    def _compute_benchmark_comparison(
        self,
        aligned_index: Optional[pd.DataFrame],
        dates: List[pd.Timestamp],
        strategy_metrics: PerformanceMetrics,
        index_symbol: Optional[str],
    ) -> BenchmarkComparison:
        """Spec section 12: 'compare against NIFTY 50 buy-and-hold.' Uses
        `aligned_index` (the benchmark already forward-filled onto `daily`'s
        own dates -- see `_align`, which only ever fills forward from past
        values, so this introduces no look-ahead) over the EXACT calendar
        window this run actually simulated."""
        if aligned_index is None or len(dates) < 2:
            return BenchmarkComparison(
                benchmark_available=False,
                symbol=index_symbol,
                note="No benchmark index data was supplied to this run (pass `index_daily=` to "
                     "Backtester.run()), or the run was too short to compare -- cannot compute a "
                     "buy-and-hold comparison.",
            )
        window = aligned_index.loc[dates[0]: dates[-1]]
        if len(window) < 2 or pd.isna(window["Close"].iloc[0]) or window["Close"].iloc[0] <= 0:
            return BenchmarkComparison(
                benchmark_available=False,
                symbol=index_symbol,
                note="Benchmark data did not overlap this run's calendar window cleanly (e.g. "
                     "entirely missing/NaN at the start) -- cannot compute a buy-and-hold "
                     "comparison for this run.",
            )
        buy_hold_return = float(window["Close"].iloc[-1] / window["Close"].iloc[0] - 1)
        years = max(len(dates) / 252, 1e-9)
        buy_hold_cagr = float((1 + buy_hold_return) ** (1 / years) - 1) if (1 + buy_hold_return) > 0 else -1.0
        return BenchmarkComparison(
            benchmark_available=True,
            symbol=index_symbol,
            buy_hold_return_pct=buy_hold_return,
            buy_hold_cagr_pct=buy_hold_cagr,
            strategy_return_pct=strategy_metrics.total_return_pct,
            strategy_cagr_pct=strategy_metrics.cagr_pct,
            outperformance_pct=strategy_metrics.total_return_pct - buy_hold_return,
            note="Plain return comparison over the same calendar window, NOT risk-adjusted -- a "
                 "strategy that sat in NO TRADE most of the time is not directly comparable to a "
                 "fully-invested buy-and-hold on this number alone. A positive outperformance_pct "
                 "here does not by itself mean the strategy is 'successful' (spec section 12): "
                 "check Sharpe/Sortino/Calmar, out-of-sample walk-forward results, and the "
                 "overfitting/robustness checks too.",
        )
