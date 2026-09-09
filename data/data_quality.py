"""
Data Quality Engine (spec section 3).

This module only MEASURES and REPORTS -- it never repairs, fills, or
fabricates data. `DataQualityChecker.check()` runs a fixed set of checks
against one symbol's OHLCV history and returns a `DataQualityReport` with a
0-1 `quality_score`, a human `status` (OK / DEGRADED / INVALID), and the
specific issues found. Every research/backtest run is expected to surface
this report (see `strategy/pipeline.py`, `backtesting/backtester.py`,
`dashboard/`) -- a run whose quality score falls below
`config.data_quality.min_quality_score_to_trade` is forced to NO TRADE
(live/paper trading) or marked INVALID (a backtest run), regardless of what
every other signal says. That enforcement lives in the callers; this module
just tells the truth about what it found.

Checks performed:
  * missing rows        -- vs. the configured trading calendar
  * duplicate timestamps
  * abnormal price jumps -- single-day |return| beyond a threshold, cross-
    checked against a caller-supplied set of known corporate-action dates
    so a real split/bonus doesn't get flagged as a data error
  * invalid OHLC values -- High < Low, High/Low inconsistent with Open/Close,
    non-positive prices
  * zero/abnormal volume
  * stale data          -- most recent bar older than the configured limit
  * timezone problems    -- naive (unattributed) timestamps
  * market holidays      -- data present on a day the calendar believes is
    a holiday (informational: the built-in NSE holiday list is deliberately
    incomplete for variable-date holidays -- see data/market_calendar.py)
  * corporate-action inconsistencies -- folded into the price-jump check
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as date_type, datetime, timezone
from typing import List, Optional, Set

import numpy as np
import pandas as pd

from data.market_calendar import NSECalendar, get_calendar


@dataclass
class DataQualityIssue:
    check: str
    severity: str   # "INFO", "WARNING", "CRITICAL"
    message: str


@dataclass
class DataQualityReport:
    symbol: str
    as_of: datetime
    issues: List[DataQualityIssue] = field(default_factory=list)
    quality_score: float = 1.0    # 0-1, 1.0 = no issues found
    status: str = "OK"            # "OK", "DEGRADED", "INVALID"
    min_quality_score_to_trade: float = 0.70

    @property
    def has_critical_issues(self) -> bool:
        return any(i.severity == "CRITICAL" for i in self.issues)

    def is_tradeable(self) -> bool:
        """The single gate callers should use: below this, the strategy
        pipeline must force NO TRADE / a backtest run must be marked
        INVALID, regardless of every other signal."""
        return (not self.has_critical_issues) and self.quality_score >= self.min_quality_score_to_trade

    def render_text(self) -> str:
        lines = [
            f"Data Quality [{self.symbol}] as of {self.as_of.isoformat()}: "
            f"{self.status} (score={self.quality_score:.2f}, threshold={self.min_quality_score_to_trade:.2f})"
        ]
        if not self.issues:
            lines.append("  No issues detected.")
        for i in self.issues:
            lines.append(f"  [{i.severity}] {i.check}: {i.message}")
        return "\n".join(lines)


class DataQualityChecker:
    def __init__(
        self,
        calendar: Optional[NSECalendar] = None,
        max_missing_row_fraction: float = 0.05,
        max_stale_data_days: int = 5,
        abnormal_daily_return_threshold: float = 0.20,
        min_volume_for_liquidity_check: float = 1.0,
        min_quality_score_to_trade: float = 0.70,
    ):
        self.calendar = calendar or get_calendar("NSE")
        self.max_missing_row_fraction = max_missing_row_fraction
        self.max_stale_data_days = max_stale_data_days
        self.abnormal_daily_return_threshold = abnormal_daily_return_threshold
        self.min_volume_for_liquidity_check = min_volume_for_liquidity_check
        self.min_quality_score_to_trade = min_quality_score_to_trade

    def check(
        self,
        symbol: str,
        daily: Optional[pd.DataFrame],
        as_of: Optional[datetime] = None,
        known_corporate_action_dates: Optional[Set[date_type]] = None,
    ) -> DataQualityReport:
        as_of = as_of or datetime.now(timezone.utc)
        issues: List[DataQualityIssue] = []
        deductions = 0.0

        if daily is None or daily.empty:
            issues.append(DataQualityIssue("missing_rows", "CRITICAL", "No data returned at all."))
            return DataQualityReport(
                symbol=symbol, as_of=as_of, issues=issues, quality_score=0.0, status="INVALID",
                min_quality_score_to_trade=self.min_quality_score_to_trade,
            )

        required = {"Open", "High", "Low", "Close", "Volume"}
        if not required.issubset(set(daily.columns)):
            issues.append(DataQualityIssue("schema", "CRITICAL", f"Missing required column(s): {required - set(daily.columns)}."))
            return DataQualityReport(
                symbol=symbol, as_of=as_of, issues=issues, quality_score=0.0, status="INVALID",
                min_quality_score_to_trade=self.min_quality_score_to_trade,
            )

        # --- duplicate timestamps -------------------------------------
        dup_count = int(daily.index.duplicated().sum())
        if dup_count:
            issues.append(DataQualityIssue(
                "duplicate_timestamps", "CRITICAL", f"{dup_count} duplicate timestamp(s) found in the raw fetch."))
            deductions += 0.3

        # --- timezone problems ------------------------------------------
        tz = getattr(daily.index, "tz", None)
        if tz is None:
            issues.append(DataQualityIssue(
                "timezone", "WARNING",
                "Timestamps are timezone-naive; treated as Asia/Kolkata wall-clock by convention, not verified."))
            deductions += 0.05

        # --- invalid OHLC relationships ----------------------------------
        o, h, l, c = daily["Open"], daily["High"], daily["Low"], daily["Close"]
        invalid_mask = (
            (h < l) | (h < o) | (h < c) | (l > o) | (l > c) |
            (o <= 0) | (h <= 0) | (l <= 0) | (c <= 0)
        )
        invalid_count = int(invalid_mask.fillna(True).sum())
        if invalid_count:
            issues.append(DataQualityIssue(
                "invalid_ohlc", "CRITICAL",
                f"{invalid_count} bar(s) with an invalid OHLC relationship (e.g. High < Low, "
                f"non-positive price) or NaN price."))
            deductions += 0.3

        # --- zero/abnormal volume -----------------------------------------
        zero_vol = int((daily["Volume"] < self.min_volume_for_liquidity_check).sum())
        if zero_vol:
            frac = zero_vol / len(daily)
            sev = "WARNING" if frac < 0.10 else "CRITICAL"
            issues.append(DataQualityIssue(
                "zero_volume", sev, f"{zero_vol} bar(s) ({frac:.1%}) with volume below "
                f"{self.min_volume_for_liquidity_check:g}."))
            deductions += 0.10 if sev == "WARNING" else 0.25

        # --- abnormal price jumps (cross-checked vs corporate actions) ---
        returns = daily["Close"].pct_change().dropna()
        jump_idx = returns[returns.abs() > self.abnormal_daily_return_threshold].index
        known_ca = known_corporate_action_dates or set()

        def _to_date(ts) -> date_type:
            return ts.date() if hasattr(ts, "date") else ts

        unexplained_jumps = [ts for ts in jump_idx if _to_date(ts) not in known_ca]
        if unexplained_jumps:
            issues.append(DataQualityIssue(
                "abnormal_price_jump", "WARNING",
                f"{len(unexplained_jumps)} day(s) with |return| > {self.abnormal_daily_return_threshold:.0%} "
                f"not matched to a supplied corporate-action date -- verify against the source before "
                f"trusting this bar (could be an unlisted split/bonus, or a genuine data error)."))
            deductions += min(0.05 * len(unexplained_jumps), 0.20)

        # --- missing rows vs. the trading calendar ------------------------
        start_d, end_d = _to_date(daily.index[0]), _to_date(daily.index[-1])
        present_dates = {_to_date(ts) for ts in daily.index}
        missing = self.calendar.missing_trading_days(present_dates, start_d, end_d)
        expected_days = len(self.calendar.trading_days_between(start_d, end_d))
        missing_fraction = (len(missing) / expected_days) if expected_days else 0.0
        if missing_fraction > self.max_missing_row_fraction:
            sev = "WARNING" if missing_fraction < 0.20 else "CRITICAL"
            issues.append(DataQualityIssue(
                "missing_rows", sev,
                f"{len(missing)}/{expected_days} expected trading day(s) missing ({missing_fraction:.1%}) "
                f"per the configured calendar."))
            deductions += min(missing_fraction, 0.4)

        # --- market-holiday mismatch (informational only) -----------------
        holiday_hits = [d for d in present_dates if not self.calendar.is_trading_day(d)]
        if holiday_hits:
            issues.append(DataQualityIssue(
                "holiday_mismatch", "INFO",
                f"{len(holiday_hits)} bar(s) fall on a date the configured calendar marks as a "
                f"non-trading day. Not scored as an error: this system's built-in NSE holiday list "
                f"only covers fixed-date national holidays (see data/market_calendar.py) and is "
                f"deliberately incomplete for variable-date holidays -- this may just mean the "
                f"calendar doesn't know about a legitimate special session or holiday."))

        # --- stale data ------------------------------------------------
        last_date = _to_date(daily.index[-1])
        as_of_date = as_of.date()
        stale_days = (as_of_date - last_date).days
        if stale_days > self.max_stale_data_days:
            sev = "WARNING" if stale_days < self.max_stale_data_days * 3 else "CRITICAL"
            issues.append(DataQualityIssue(
                "stale_data", sev, f"Most recent bar is {stale_days} calendar day(s) old (as of {as_of_date})."))
            deductions += min(0.02 * stale_days, 0.3)

        quality_score = float(np.clip(1.0 - deductions, 0.0, 1.0))
        has_critical = any(i.severity == "CRITICAL" for i in issues)
        if has_critical:
            status = "INVALID"
        elif quality_score < self.min_quality_score_to_trade or any(i.severity == "WARNING" for i in issues):
            status = "DEGRADED"
        else:
            status = "OK"

        return DataQualityReport(
            symbol=symbol, as_of=as_of, issues=issues, quality_score=quality_score, status=status,
            min_quality_score_to_trade=self.min_quality_score_to_trade,
        )
