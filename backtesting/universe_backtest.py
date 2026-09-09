"""
Cross-stock robustness reporting (spec Part 22): "do not declare success
because one stock performs well." `backtesting/backtester.py`'s
`Backtester.run()` backtests ONE symbol at a time by design (its own
docstring says multi-symbol orchestration is the caller's job) -- this
module is that caller: loop the configured universe, run each symbol
through the SAME `Backtester`/`StrategyPipeline` paper trading uses, and
aggregate median/percentage-positive statistics across the whole batch
instead of ever reporting a single symbol's numbers as if they were the
strategy's numbers.

NO TRADE is not failure. A symbol where the strategy correctly found
nothing worth trading has zero trades, and `backtesting.metrics.
compute_metrics` reports that as `profit_factor=0.0` (no wins, no losses)
-- folding that into "% of symbols with PF > 1" would misrepresent a
correct NO-TRADE outcome as a losing one. Aggregate statistics here are
computed ONLY over symbols that took at least one trade; zero-trade and
data-invalid symbols are counted and reported separately, never hidden.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from backtesting.backtester import Backtester, BacktestConfig
from config.settings import Config
from data.market_data import DataUnavailableError, MarketDataProvider

logger = logging.getLogger(__name__)


@dataclass
class SymbolBacktestSummary:
    symbol: str
    sector: Optional[str]
    is_valid: bool           # False = data fetch failed or the run-level data-quality gate failed
    num_trades: int
    profit_factor: float = 0.0
    expectancy: float = 0.0
    cagr_pct: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate_pct: float = 0.0
    note: str = ""


@dataclass
class SectorRollup:
    sector: str
    symbols_traded: int
    median_profit_factor: float
    pct_profit_factor_above_1: float
    pct_positive_expectancy: float


@dataclass
class UniverseBacktestReport:
    universe_size: int
    symbols_traded: int      # took >= 1 trade -- the ONLY population aggregates are computed over
    no_trade_count: int      # valid run, correctly took zero trades -- not a failure, see module docstring
    invalid_count: int       # data fetch failed or data-quality gate failed
    summaries: List[SymbolBacktestSummary] = field(default_factory=list)
    median_profit_factor: float = 0.0
    pct_profit_factor_above_1: float = 0.0
    pct_positive_expectancy: float = 0.0
    pct_positive_cagr: float = 0.0
    pct_positive_sharpe: float = 0.0
    sector_rollups: Dict[str, SectorRollup] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Cross-stock robustness report ({self.universe_size} symbol(s) requested):",
            f"  {self.symbols_traded} took >= 1 trade, {self.no_trade_count} correctly took zero trades "
            f"(NOT a failure -- NO TRADE is this system's default outcome), {self.invalid_count} "
            f"had unusable/unfetchable data.",
        ]
        if self.symbols_traded == 0:
            lines.append(
                "  No symbol took a single trade over this period -- nothing to aggregate. This is "
                "an honest, valid outcome (see the per-symbol notes above), not an error."
            )
            return "\n".join(lines)
        lines += [
            f"  Median profit factor (traded symbols only): {self.median_profit_factor:.2f}",
            f"  % of traded symbols with profit factor > 1: {self.pct_profit_factor_above_1:.1%}",
            f"  % of traded symbols with positive expectancy: {self.pct_positive_expectancy:.1%}",
            f"  % of traded symbols with positive CAGR: {self.pct_positive_cagr:.1%}",
            f"  % of traded symbols with positive Sharpe: {self.pct_positive_sharpe:.1%}",
        ]
        if self.sector_rollups:
            lines.append("  By sector (traded symbols only):")
            for sector, r in sorted(self.sector_rollups.items()):
                lines.append(
                    f"    {sector:<15} n={r.symbols_traded:<3} median_PF={r.median_profit_factor:.2f} "
                    f"%PF>1={r.pct_profit_factor_above_1:.1%} %pos_expectancy={r.pct_positive_expectancy:.1%}"
                )
        lines.append(
            "  Disclaimer: cross-stock statistics describe THIS universe over THIS period only -- "
            "not a guarantee of future or out-of-universe performance."
        )
        return "\n".join(lines)


def _aggregate(summaries: List[SymbolBacktestSummary]) -> Dict[str, float]:
    pfs = [s.profit_factor for s in summaries]
    return {
        "median_profit_factor": float(np.median(pfs)) if pfs else 0.0,
        "pct_profit_factor_above_1": float(np.mean([1.0 if pf > 1 else 0.0 for pf in pfs])) if pfs else 0.0,
        "pct_positive_expectancy": float(np.mean([1.0 if s.expectancy > 0 else 0.0 for s in summaries])) if summaries else 0.0,
        "pct_positive_cagr": float(np.mean([1.0 if s.cagr_pct > 0 else 0.0 for s in summaries])) if summaries else 0.0,
        "pct_positive_sharpe": float(np.mean([1.0 if s.sharpe_ratio > 0 else 0.0 for s in summaries])) if summaries else 0.0,
    }


def run_universe_backtest(
    config: Config,
    symbols: List[str],
    period: str = "5y",
    sector_map: Optional[Dict[str, str]] = None,
    market_data: Optional[MarketDataProvider] = None,
    backtest_config: Optional[BacktestConfig] = None,
    macro_daily: Optional[Dict[str, pd.DataFrame]] = None,
) -> UniverseBacktestReport:
    """
    Runs `Backtester.run()` once per symbol (fault-isolated -- one symbol's
    fetch/analysis failure never aborts the batch, same principle as
    `paper_trading/scanner.py`'s `UniverseScanner`) and aggregates. Never
    optimizes on or special-cases any individual symbol (spec Part 21/22).
    """
    md = market_data or MarketDataProvider()
    sector_map = sector_map or {}

    index_daily = None
    if config.universe.index_symbol:
        try:
            index_daily = md.get_daily(config.universe.index_symbol, period=period)
        except DataUnavailableError as exc:
            logger.warning("Benchmark %s unavailable: %s -- market-condition scoring UNKNOWN this run.",
                            config.universe.index_symbol, exc)

    # Batched + rate-limit-paced (spec section 19's same principle applied
    # here as paper_trading/scanner.py's UniverseScanner already applies to
    # the live-scan path) -- a NIFTY-50-sized universe fetched back-to-back
    # with zero pacing risks tripping a free/shared data vendor's rate limit.
    batch = md.get_daily_batch(
        symbols, period=period,
        batch_size=config.universe.scan_batch_size,
        delay_between_batches_seconds=config.universe.scan_batch_delay_seconds,
    )

    summaries: List[SymbolBacktestSummary] = []
    for symbol in symbols:
        sector = sector_map.get(symbol)
        daily = batch.data.get(symbol)
        if daily is None:
            exc_msg = batch.errors.get(symbol, "no data returned")
            logger.warning("Skipping %s in universe backtest: %s", symbol, exc_msg)
            summaries.append(SymbolBacktestSummary(
                symbol=symbol, sector=sector, is_valid=False, num_trades=0, note=f"data unavailable: {exc_msg}",
            ))
            continue

        try:
            backtester = Backtester(config=config, backtest_config=backtest_config)
            result = backtester.run(symbol, daily, index_daily=index_daily, sector=sector,
                                     index_symbol=config.universe.index_symbol, macro_daily=macro_daily)
        except Exception as exc:
            logger.warning("Backtest failed for %s (skipping, not aborting the batch): %s", symbol, exc)
            summaries.append(SymbolBacktestSummary(
                symbol=symbol, sector=sector, is_valid=False, num_trades=0, note=f"backtest error: {exc}",
            ))
            continue

        if not result.is_valid:
            summaries.append(SymbolBacktestSummary(
                symbol=symbol, sector=sector, is_valid=False, num_trades=0,
                note="data-quality gate failed for the run -- nothing simulated",
            ))
            continue

        m = result.metrics
        summaries.append(SymbolBacktestSummary(
            symbol=symbol, sector=sector, is_valid=True, num_trades=m.num_trades,
            profit_factor=m.profit_factor, expectancy=m.expectancy, cagr_pct=m.cagr_pct,
            sharpe_ratio=m.sharpe_ratio, max_drawdown_pct=m.max_drawdown_pct, win_rate_pct=m.win_rate_pct,
            note="" if m.num_trades > 0 else "valid run, correctly took zero trades",
        ))

    traded = [s for s in summaries if s.is_valid and s.num_trades > 0]
    no_trade = [s for s in summaries if s.is_valid and s.num_trades == 0]
    invalid = [s for s in summaries if not s.is_valid]

    agg = _aggregate(traded)

    sector_rollups: Dict[str, SectorRollup] = {}
    sectors = {s.sector or "UNSPECIFIED" for s in traded}
    for sector in sectors:
        sector_traded = [s for s in traded if (s.sector or "UNSPECIFIED") == sector]
        sector_agg = _aggregate(sector_traded)
        sector_rollups[sector] = SectorRollup(
            sector=sector, symbols_traded=len(sector_traded),
            median_profit_factor=sector_agg["median_profit_factor"],
            pct_profit_factor_above_1=sector_agg["pct_profit_factor_above_1"],
            pct_positive_expectancy=sector_agg["pct_positive_expectancy"],
        )

    return UniverseBacktestReport(
        universe_size=len(symbols), symbols_traded=len(traded), no_trade_count=len(no_trade),
        invalid_count=len(invalid), summaries=summaries,
        median_profit_factor=agg["median_profit_factor"],
        pct_profit_factor_above_1=agg["pct_profit_factor_above_1"],
        pct_positive_expectancy=agg["pct_positive_expectancy"],
        pct_positive_cagr=agg["pct_positive_cagr"],
        pct_positive_sharpe=agg["pct_positive_sharpe"],
        sector_rollups=sector_rollups,
    )
