"""
Small-account survival simulation (spec Part 25) -- explicitly called a
major acceptance criterion: does the strategy remain economically viable,
and does the ACCOUNT survive, at realistic small starting balances, not
just at the institutional-scale default?

Reuses `backtesting.backtester.Backtester` (unchanged) once per symbol per
capital level, and `backtesting.monte_carlo.run_monte_carlo` (unchanged)
to turn the resulting pool of realized trade P&Ls into a survival rate /
probability-of-ruin / drawdown-distribution read for that capital level.

Known approximations, stated here rather than hidden:
  * Pooling trades ACROSS the whole scanned universe at one capital level
    (rather than one symbol at a time) treats them as an independent,
    resampleable population -- spec Part 25 asks for the ACCOUNT's
    survival, not a per-stock one, but real trades across different NSE
    symbols are not fully independent (correlated market-wide moves exist).
  * Each symbol's `Backtester.run()` sizes its own positions as if that
    symbol ALONE had the full capital available at this level -- pooling
    those trades afterward as if they occurred sequentially in one
    shared-capital account does not model a real account holding several
    concurrent positions, which would size each one smaller. Modeling true
    shared-capital, concurrent-position sizing is a larger feature this
    module does not attempt.
Read `probability_of_ruin` as an approximate, directional read of
small-account risk, not a precise guarantee.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from backtesting.backtester import Backtester, BacktestConfig
from backtesting.monte_carlo import run_monte_carlo
from config.settings import Config
from data.market_data import DataUnavailableError, MarketDataProvider

logger = logging.getLogger(__name__)

# Spec Part 25's exact list.
SMALL_ACCOUNT_CAPITAL_LEVELS: List[float] = [100, 500, 1_000, 5_000, 10_000, 25_000, 50_000, 100_000]

INSUFFICIENT_CAPITAL_VERDICT = "INSUFFICIENT CAPITAL FOR THIS MARKET/INSTRUMENT"


@dataclass
class CapitalLevelSurvival:
    capital: float
    symbols_with_any_trade: int
    total_trades: int
    survival_rate: float = 0.0            # 1 - probability_of_ruin
    probability_of_ruin: float = 0.0
    median_final_equity: float = 0.0
    median_max_drawdown_pct: float = 0.0
    median_losing_streak: float = 0.0
    avg_transaction_cost_pct_of_capital: float = 0.0
    verdict: str = ""                     # INSUFFICIENT_CAPITAL_VERDICT, or "" (no other level is
                                           # editorialized as "success" -- see module docstring)


@dataclass
class SmallAccountSurvivalReport:
    levels: List[CapitalLevelSurvival] = field(default_factory=list)

    def summary(self) -> str:
        lines = ["Small-account survival simulation (spec Part 25):"]
        header = (
            f"  {'CAPITAL':>10} {'TRADES':>8} {'SURVIVAL':>10} {'P(RUIN)':>9} {'MED_DD':>8} "
            f"{'MED_STREAK':>10} {'MED_EQUITY':>12} {'AVG_COST%':>10}  VERDICT"
        )
        lines.append(header)
        for lvl in self.levels:
            lines.append(
                f"  {lvl.capital:>10,.0f} {lvl.total_trades:>8} {lvl.survival_rate:>10.1%} "
                f"{lvl.probability_of_ruin:>9.1%} {lvl.median_max_drawdown_pct:>8.1%} "
                f"{lvl.median_losing_streak:>10.1f} {lvl.median_final_equity:>12,.0f} "
                f"{lvl.avg_transaction_cost_pct_of_capital:>10.2%}  {lvl.verdict}"
            )
        lines.append(
            "  Approximation: trades are pooled across the whole scanned universe per capital level "
            "(assumes independence across symbols, which real cross-market moves violate somewhat), "
            "and each symbol's trades were position-sized independently as if that symbol alone had "
            "the full capital available, then pooled as if sequential in one shared-capital account "
            "(a real account holding several positions at once would size each smaller) -- read "
            "probability_of_ruin as directional, not exact. See module docstring."
        )
        return "\n".join(lines)


def run_small_account_survival(
    config: Config,
    symbols: List[str],
    capital_levels: Optional[List[float]] = None,
    period: str = "5y",
    market_data: Optional[MarketDataProvider] = None,
    num_simulations: int = 2000,
    seed: Optional[int] = 42,
    macro_daily: Optional[Dict[str, pd.DataFrame]] = None,
) -> SmallAccountSurvivalReport:
    capital_levels = list(capital_levels) if capital_levels is not None else list(SMALL_ACCOUNT_CAPITAL_LEVELS)
    md = market_data or MarketDataProvider()

    index_daily = None
    if config.universe.index_symbol:
        try:
            index_daily = md.get_daily(config.universe.index_symbol, period=period)
        except DataUnavailableError as exc:
            logger.warning("Benchmark %s unavailable: %s", config.universe.index_symbol, exc)

    # Fetch each symbol's OHLCV ONCE -- reused across every capital level
    # (position sizing/trade count differ by capital; the underlying price
    # history doesn't). Batched + rate-limit-paced the same way
    # run_universe_backtest fetches its universe (see
    # MarketDataProvider.get_daily_batch); batching here is enough since
    # this loop runs once regardless of how many capital levels follow --
    # there's no per-capital-level refetch to protect.
    batch = md.get_daily_batch(
        symbols, period=period,
        batch_size=config.universe.scan_batch_size,
        delay_between_batches_seconds=config.universe.scan_batch_delay_seconds,
    )
    for symbol, exc_msg in batch.errors.items():
        logger.warning("Skipping %s in small-account survival: %s", symbol, exc_msg)
    data_by_symbol: Dict[str, pd.DataFrame] = batch.data

    levels: List[CapitalLevelSurvival] = []
    for capital in capital_levels:
        bt_config = BacktestConfig(
            brokerage_pct=config.backtesting.brokerage_pct, taxes_pct=config.backtesting.taxes_pct,
            slippage_pct=config.backtesting.slippage_pct, bid_ask_spread_pct=config.backtesting.bid_ask_spread_pct,
            initial_capital=capital,
        )
        pooled_pnls: List[float] = []
        cost_pcts: List[float] = []
        symbols_with_trade = 0

        for symbol, daily in data_by_symbol.items():
            try:
                backtester = Backtester(config=config, backtest_config=bt_config)
                result = backtester.run(symbol, daily, index_daily=index_daily,
                                         index_symbol=config.universe.index_symbol, macro_daily=macro_daily)
            except Exception as exc:
                logger.warning("Backtest failed for %s at capital=%s: %s", symbol, capital, exc)
                continue
            if not result.is_valid or not result.trades:
                continue
            symbols_with_trade += 1
            for t in result.trades:
                pooled_pnls.append(t.net_pnl)
                if capital > 0:
                    cost_pcts.append(t.costs / capital)

        if not pooled_pnls:
            levels.append(CapitalLevelSurvival(
                capital=capital, symbols_with_any_trade=0, total_trades=0,
                survival_rate=float("nan"), probability_of_ruin=float("nan"),
                verdict=INSUFFICIENT_CAPITAL_VERDICT,
            ))
            continue

        mc = run_monte_carlo(pooled_pnls, initial_capital=capital, num_simulations=num_simulations,
                              method="bootstrap", seed=seed)
        p50 = mc.percentiles.get(50, {})
        levels.append(CapitalLevelSurvival(
            capital=capital, symbols_with_any_trade=symbols_with_trade, total_trades=len(pooled_pnls),
            survival_rate=1.0 - mc.probability_of_ruin, probability_of_ruin=mc.probability_of_ruin,
            median_final_equity=p50.get("final_equity", float("nan")),
            median_max_drawdown_pct=p50.get("max_drawdown_pct", float("nan")),
            median_losing_streak=p50.get("losing_streak", float("nan")),
            avg_transaction_cost_pct_of_capital=float(np.mean(cost_pcts)) if cost_pcts else 0.0,
            verdict="",
        ))

    # Distinguish "genuinely can't afford this instrument" from "no
    # qualifying trade signal fired at ANY capital this period" -- the
    # same conflation caught in account-check's affordability reporting
    # (spec Part 10/25 vs. Part 17 "no trade is a valid decision"). Only
    # reclassify when there's actual comparative evidence: at least one
    # tested capital level at or above the small-account threshold (spec
    # Part 10-14's own INR 25,000 line, config.risk.small_account_capital_
    # threshold -- reused here rather than inventing a second one) ALSO
    # took zero trades. A single small level tested alone (no large
    # reference point) keeps the direct INSUFFICIENT_CAPITAL_VERDICT --
    # there's no evidence either way to say otherwise.
    if levels:
        threshold = config.risk.small_account_capital_threshold
        large_reference_levels = [lvl for lvl in levels if lvl.capital >= threshold]
        if large_reference_levels and all(lvl.total_trades == 0 for lvl in large_reference_levels):
            for lvl in levels:
                if lvl.verdict == INSUFFICIENT_CAPITAL_VERDICT:
                    lvl.verdict = (
                        "No qualifying trade signal fired for this universe/period even at "
                        f">= INR {threshold:,.0f} capital -- not a capital constraint."
                    )

    return SmallAccountSurvivalReport(levels=levels)
