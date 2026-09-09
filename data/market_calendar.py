"""
NSE (India) trading calendar and timezone handling.

Real, verifiable rules baked in by default:
  * NSE equity trading runs Monday-Friday (no assumption of US market hours
    or a US-style calendar anywhere in this module).
  * A handful of national holidays fall on a FIXED Gregorian date every
    year and NSE is reliably closed: Republic Day (Jan 26), Independence
    Day (Aug 15), Gandhi Jayanti (Oct 2).

What this module deliberately does NOT do: invent exact dates for the
variable-date holidays (Diwali/Muhurat trading, Holi, Good Friday, Eid,
Ram Navami, Ganesh Chaturthi, Maharashtra Day, Ambedkar Jayanti, etc.).
Those shift every year (many follow lunar/regional calendars) and NSE
publishes the authoritative list each year at
https://www.nseindia.com/resources/exchange-communication-holidays --
hard-coding guessed dates here would silently corrupt data-quality checks
and calendar-aware backtests with WRONG holidays, which is worse than
being incomplete about them. Instead:

  * `NSECalendar` ships correct-by-construction weekend + fixed-holiday
    handling out of the box.
  * Pass `extra_holidays=` (a set of `date`) or use `NSECalendar.from_csv(path)`
    to load a full, accurate holiday list you've copied from NSE's own
    circular -- every `DataQualityReport` holiday check reports which
    calendar (fixed-only vs. fixed+extra) it evaluated against, so a false
    positive from an incomplete built-in list is visible, not silent.

All timestamps in this system default to `Asia/Kolkata` (IST, UTC+5:30) --
never assume US market hours or US Eastern time anywhere downstream.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Iterable, List, Optional, Set

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - Python < 3.9 fallback, not expected here
    ZoneInfo = None  # type: ignore

NSE_TIMEZONE = "Asia/Kolkata"

# (month, day) pairs -- fixed national holidays, correct for every year.
FIXED_NATIONAL_HOLIDAYS_MMDD = [(1, 26), (8, 15), (10, 2)]

# NSE cash-market session, IST. Used for "was this decision timestamp during
# market hours" checks, not for intraday bar generation.
MARKET_OPEN_TIME = (9, 15)
MARKET_CLOSE_TIME = (15, 30)


def kolkata_tz():
    """Returns the Asia/Kolkata tzinfo, or None if zoneinfo data isn't
    installed (callers should fall back to naive/UTC handling and log a
    warning rather than silently assuming a different zone)."""
    if ZoneInfo is None:
        return None
    try:
        return ZoneInfo(NSE_TIMEZONE)
    except Exception:
        return None


def localize_to_kolkata(ts: datetime) -> datetime:
    """Attaches (or converts to) Asia/Kolkata. A naive datetime is assumed
    to already represent IST wall-clock time (the common case for daily bar
    timestamps from an NSE-focused data vendor) and is simply tagged, not
    shifted -- shifting a naive timestamp would silently assume a timezone
    that was never stated."""
    tz = kolkata_tz()
    if tz is None:
        return ts
    if ts.tzinfo is None:
        return ts.replace(tzinfo=tz)
    return ts.astimezone(tz)


@dataclass
class NSECalendar:
    """Trading-day calendar for NSE. Weekends + the three fixed national
    holidays are always enforced (when `include_fixed_national_holidays` is
    True, the default); supply `extra_holidays` for full accuracy against
    NSE's actual published holiday list. See `get_calendar()` for a
    factory that also serves a correct, honest weekend-only calendar for a
    non-NSE `system.trading_calendar` setting, instead of guessing another
    market's holidays."""

    extra_holidays: Set[date] = field(default_factory=set)
    timezone: str = NSE_TIMEZONE
    include_fixed_national_holidays: bool = True

    def _is_fixed_holiday(self, d: date) -> bool:
        return self.include_fixed_national_holidays and (d.month, d.day) in FIXED_NATIONAL_HOLIDAYS_MMDD

    def is_weekend(self, d: date) -> bool:
        return d.weekday() >= 5  # Saturday=5, Sunday=6

    def is_holiday(self, d: date) -> bool:
        return self._is_fixed_holiday(d) or d in self.extra_holidays

    def is_trading_day(self, d: date) -> bool:
        return not self.is_weekend(d) and not self.is_holiday(d)

    def holiday_reason(self, d: date) -> Optional[str]:
        """Explains WHY a date is not a trading day, or None if it is one."""
        if self.is_weekend(d):
            return "weekend"
        if self._is_fixed_holiday(d):
            return "fixed national holiday"
        if d in self.extra_holidays:
            return "configured NSE holiday"
        return None

    def next_trading_day(self, d: date) -> date:
        nxt = d + timedelta(days=1)
        while not self.is_trading_day(nxt):
            nxt += timedelta(days=1)
        return nxt

    def previous_trading_day(self, d: date) -> date:
        prev = d - timedelta(days=1)
        while not self.is_trading_day(prev):
            prev -= timedelta(days=1)
        return prev

    def trading_days_between(self, start: date, end: date) -> List[date]:
        if end < start:
            return []
        days = []
        cur = start
        while cur <= end:
            if self.is_trading_day(cur):
                days.append(cur)
            cur += timedelta(days=1)
        return days

    def is_market_hours(self, ts: datetime) -> bool:
        """Whether `ts` (assumed/attached Asia/Kolkata) falls within the NSE
        cash session on a trading day. Used to sanity-check "as of" decision
        timestamps for live/paper trading -- never assumes US market hours."""
        local = localize_to_kolkata(ts)
        d = local.date()
        if not self.is_trading_day(d):
            return False
        open_t = local.replace(hour=MARKET_OPEN_TIME[0], minute=MARKET_OPEN_TIME[1], second=0, microsecond=0)
        close_t = local.replace(hour=MARKET_CLOSE_TIME[0], minute=MARKET_CLOSE_TIME[1], second=0, microsecond=0)
        return open_t <= local <= close_t

    @classmethod
    def from_csv(cls, path: str, timezone: str = NSE_TIMEZONE) -> "NSECalendar":
        """Loads extra holidays from a CSV with a `date` column (YYYY-MM-DD),
        e.g. one you copied from NSE's published holiday circular. Only
        adds to the built-in fixed holidays -- never replaces them."""
        holidays: Set[date] = set()
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            key = "date" if reader.fieldnames and "date" in reader.fieldnames else reader.fieldnames[0]
            for row in reader:
                raw = row[key].strip()
                if raw:
                    holidays.add(datetime.strptime(raw, "%Y-%m-%d").date())
        return cls(extra_holidays=holidays, timezone=timezone)

    def missing_trading_days(self, present_dates: Iterable[date], start: date, end: date) -> List[date]:
        """Given the dates actually present in a market-data series, returns
        the trading days in [start, end] that are MISSING from it -- the
        core "missing rows" check for DataQualityReport."""
        present = set(present_dates)
        return [d for d in self.trading_days_between(start, end) if d not in present]


def get_calendar(name: str, extra_holidays: Optional[Iterable[date]] = None) -> NSECalendar:
    """Factory keyed by `system.trading_calendar` (or `system.exchange`).
    Only "NSE" gets India's fixed national holidays baked in -- any other
    value (including "generic"/"US"/"BSE") gets a correct, honest
    weekend-only calendar rather than a guessed set of holidays for a
    market this module doesn't actually know. BSE's trading calendar is, in
    practice, the same as NSE's for equity cash trading; pass "NSE" for
    `system.trading_calendar` for a BSE-only deployment too, or supply your
    own `extra_holidays`."""
    holidays: Set[date] = set(extra_holidays) if extra_holidays else set()
    include_fixed = (name or "").upper() == "NSE"
    return NSECalendar(extra_holidays=holidays, include_fixed_national_holidays=include_fixed)
