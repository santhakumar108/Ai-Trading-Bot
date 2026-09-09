"""
Performance metrics for backtests and paper-trading history.

Every function takes plain lists/arrays (trade P&Ls, an equity curve) so it
can be reused by the backtester, the walk-forward validator, and the paper
trading engine without any coupling between them.

Spec section 12 ("Performance analysis") is explicit that a strategy must
NOT be called successful just because total return is positive -- this
module's job is to compute the numbers honestly, not to render a verdict.
Nothing here scores or labels a run as "good"; `backtesting/health_report.py`
(overfitting/robustness checks) and the caller's own judgment are where any
verdict belongs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class PerformanceMetrics:
    total_return_pct: float
    cagr_pct: float
    annualized_volatility_pct: float   # std of period returns, annualized -- how bumpy the ride was,
                                        # independent of Sharpe (which divides by this same number)
    win_rate_pct: float
    average_win: float
    average_loss: float
    profit_factor: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float                # CAGR / |max_drawdown_pct| -- return per unit of the worst
                                        # peak-to-trough loss actually realized, annualized (unlike
                                        # risk_adjusted_return below, which is not time-normalized)
    expectancy: float
    num_trades: int
    longest_losing_streak: int
    recovery_factor: float             # net profit (currency) / worst peak-to-trough drawdown
                                        # (currency) -- how many times over the strategy "earned
                                        # back" its own worst drawdown
    risk_adjusted_return: float        # total_return_pct / |max_drawdown_pct|, guarded (kept for
                                        # backward compatibility; NOT annualized -- prefer
                                        # calmar_ratio for a like-for-like comparison across
                                        # backtests of different lengths)
    monthly_returns: Dict[str, float] = field(default_factory=dict)   # "YYYY-MM" -> period return
    yearly_returns: Dict[str, float] = field(default_factory=dict)    # "YYYY" -> period return

    def as_dict(self) -> dict:
        out = {}
        for k, v in self.__dict__.items():
            if isinstance(v, float):
                out[k] = round(v, 4)
            elif isinstance(v, dict):
                out[k] = {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()}
            else:
                out[k] = v
        return out


def max_drawdown(equity_curve: Sequence[float]) -> float:
    if len(equity_curve) == 0:
        return 0.0
    curve = np.array(equity_curve, dtype=float)
    running_max = np.maximum.accumulate(curve)
    drawdowns = (curve - running_max) / np.where(running_max == 0, 1, running_max)
    return float(drawdowns.min())  # negative number, e.g. -0.18 = -18%


def max_drawdown_currency(equity_curve: Sequence[float]) -> float:
    """Worst peak-to-trough loss in absolute currency terms (a positive
    number, or 0.0 if the curve never fell below a prior peak)."""
    if len(equity_curve) == 0:
        return 0.0
    curve = np.array(equity_curve, dtype=float)
    running_max = np.maximum.accumulate(curve)
    return float((running_max - curve).max())


def longest_losing_streak(trade_pnls: Sequence[float]) -> int:
    longest = current = 0
    for pnl in trade_pnls:
        if pnl < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def sharpe_ratio(period_returns: Sequence[float], periods_per_year: int = 252, risk_free_rate: float = 0.0) -> float:
    r = np.array(period_returns, dtype=float)
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    excess = r - (risk_free_rate / periods_per_year)
    return float(np.mean(excess) / np.std(excess, ddof=1) * np.sqrt(periods_per_year))


def sortino_ratio(period_returns: Sequence[float], periods_per_year: int = 252, risk_free_rate: float = 0.0) -> float:
    r = np.array(period_returns, dtype=float)
    if len(r) < 2:
        return 0.0
    excess = r - (risk_free_rate / periods_per_year)
    downside = excess[excess < 0]
    downside_std = downside.std(ddof=1) if len(downside) > 1 else 0.0
    if downside_std == 0:
        return 0.0
    return float(np.mean(excess) / downside_std * np.sqrt(periods_per_year))


def annualized_volatility(period_returns: Sequence[float], periods_per_year: int = 252) -> float:
    r = np.array(period_returns, dtype=float)
    if len(r) < 2:
        return 0.0
    return float(np.std(r, ddof=1) * np.sqrt(periods_per_year))


def _periodic_returns(
    equity_curve: Sequence[float], dates: Optional[Sequence], initial_capital: float, period: str = "month",
) -> Dict[str, float]:
    """Buckets an equity curve by calendar month/year and returns the simple
    return of each bucket relative to the PRIOR bucket's closing equity (the
    very first bucket is measured against `initial_capital`). Pure Python
    (no pandas resample) so this doesn't depend on a particular pandas
    version's resample-alias spelling ('M' vs 'ME')."""
    if not equity_curve or not dates or len(equity_curve) != len(dates):
        return {}
    pairs = sorted(zip(dates, equity_curve), key=lambda p: p[0])
    period_end_value: Dict[str, float] = {}
    order: List[str] = []
    for d, v in pairs:
        ts = pd.Timestamp(d)
        key = f"{ts.year:04d}-{ts.month:02d}" if period == "month" else f"{ts.year:04d}"
        if key not in period_end_value:
            order.append(key)
        period_end_value[key] = float(v)  # overwritten each time -> last value seen per key, since
                                           # `pairs` is chronologically sorted
    out: Dict[str, float] = {}
    baseline = initial_capital
    for key in order:
        val = period_end_value[key]
        out[key] = float((val - baseline) / baseline) if baseline else 0.0
        baseline = val
    return out


def monthly_returns(equity_curve: Sequence[float], dates: Optional[Sequence], initial_capital: float) -> Dict[str, float]:
    return _periodic_returns(equity_curve, dates, initial_capital, period="month")


def yearly_returns(equity_curve: Sequence[float], dates: Optional[Sequence], initial_capital: float) -> Dict[str, float]:
    return _periodic_returns(equity_curve, dates, initial_capital, period="year")


def compute_metrics(
    trade_pnls: Sequence[float],
    equity_curve: Sequence[float],
    period_returns: Sequence[float],
    initial_capital: float,
    num_trading_periods: int,
    periods_per_year: int = 252,
    dates: Optional[Sequence] = None,
) -> PerformanceMetrics:
    """
    `dates`, if supplied (one per `equity_curve` entry), enables the
    monthly/yearly return breakdown (spec section 12). It's optional and
    defaults to None because this function is also used to compute
    per-regime metrics (`Backtester.run`'s `metrics_by_regime`), where the
    "dates" are a synthetic cumulative-P&L series with gaps between them --
    a monthly/yearly breakdown of that series would be misleading, so it's
    correctly left empty ({}) in that case rather than computed from
    meaningless buckets.
    """
    pnls = np.array(trade_pnls, dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    total_return_pct = 0.0
    if initial_capital > 0 and len(equity_curve):
        total_return_pct = (equity_curve[-1] - initial_capital) / initial_capital

    years = max(num_trading_periods / periods_per_year, 1e-9)
    cagr_pct = 0.0
    if initial_capital > 0 and len(equity_curve) and equity_curve[-1] > 0:
        cagr_pct = (equity_curve[-1] / initial_capital) ** (1 / years) - 1

    win_rate_pct = float(len(wins) / len(pnls)) if len(pnls) else 0.0
    average_win = float(wins.mean()) if len(wins) else 0.0
    average_loss = float(losses.mean()) if len(losses) else 0.0
    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(-losses.sum()) if len(losses) else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)

    mdd = max_drawdown(equity_curve)
    mdd_currency = max_drawdown_currency(equity_curve)
    sharpe = sharpe_ratio(period_returns, periods_per_year)
    sortino = sortino_ratio(period_returns, periods_per_year)
    ann_vol = annualized_volatility(period_returns, periods_per_year)
    calmar = (cagr_pct / abs(mdd)) if mdd != 0 else 0.0

    expectancy = float(pnls.mean()) if len(pnls) else 0.0
    streak = longest_losing_streak(pnls)

    risk_adjusted_return = (total_return_pct / abs(mdd)) if mdd != 0 else 0.0

    net_profit_currency = (equity_curve[-1] - initial_capital) if len(equity_curve) else 0.0
    if mdd_currency > 0:
        recovery = net_profit_currency / mdd_currency
    else:
        recovery = float("inf") if net_profit_currency > 0 else 0.0

    monthly = _periodic_returns(equity_curve, dates, initial_capital, period="month")
    yearly = _periodic_returns(equity_curve, dates, initial_capital, period="year")

    return PerformanceMetrics(
        total_return_pct=total_return_pct,
        cagr_pct=cagr_pct,
        annualized_volatility_pct=ann_vol,
        win_rate_pct=win_rate_pct,
        average_win=average_win,
        average_loss=average_loss,
        profit_factor=profit_factor,
        max_drawdown_pct=mdd,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        expectancy=expectancy,
        num_trades=int(len(pnls)),
        longest_losing_streak=streak,
        recovery_factor=recovery,
        risk_adjusted_return=risk_adjusted_return,
        monthly_returns=monthly,
        yearly_returns=yearly,
    )
