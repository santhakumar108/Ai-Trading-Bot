"""
Point-in-time data model (mandatory per spec section 8).

Every information item the strategy consumes -- a price bar, a technical
indicator, a fundamentals snapshot, a news item, a sentiment reading, a
corporate action -- has two distinct timestamps:

  * `data_timestamp`  -- what the data DESCRIBES (a trading day, a fiscal
    quarter-end, the moment an event occurred).
  * `available_timestamp` -- when that information was actually, publicly
    knowable. For a daily bar this is the bar's own session-close time.
    For a quarterly result, this is the public filing/announcement date,
    which is essentially always LATER than the quarter's `data_timestamp`
    (a Q1 quarter ends Jun 30 but is typically not filed until weeks
    later). For a news article, it's (approximately) `published_at`.

The one rule the whole system must respect, everywhere:

    available_timestamp <= decision_timestamp

This module is the single, reusable primitive for that rule so it can be
tested directly rather than merely "trusted" to hold inside each provider's
bespoke filtering logic. `backtesting/historical_providers.py`,
`data/market_data.py`, `news/news_analysis.py`, and
`sentiment/social_sentiment.py` all express this same rule in their own
terms (an `as_of` cutoff, a `published_at`/`created_at` filter, an
`.iloc[:i+1]` slice) -- this module gives it one explicit name and one
place `tests/test_point_in_time.py` can test against directly, and a
utility other providers can build on rather than reimplementing filtering
by hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Generic, Iterable, List, Optional, TypeVar

T = TypeVar("T")


class LookaheadViolationError(RuntimeError):
    """Raised when a caller tries to force the use of information before it
    was actually available. This is a hard bug wherever it happens -- it
    silently invalidates any backtest or live decision that touches it --
    so it is an exception, not a value callers might ignore."""


@dataclass(frozen=True)
class PointInTimeValue(Generic[T]):
    """Wraps one piece of information with both timestamps required by the
    point-in-time rule."""

    value: T
    data_timestamp: datetime
    available_timestamp: datetime

    def __post_init__(self):
        if self.available_timestamp < self.data_timestamp:
            # Not necessarily impossible in every domain, but for every data
            # type this system handles (prices, fundamentals, news,
            # sentiment, corporate actions) information becomes knowable no
            # earlier than the event/period it describes. This almost
            # always indicates a bug in the caller's timestamp assignment.
            raise ValueError(
                f"available_timestamp ({self.available_timestamp}) is before "
                f"data_timestamp ({self.data_timestamp}) -- that would mean "
                f"this was known before it happened."
            )

    def is_available_by(self, decision_timestamp: datetime) -> bool:
        return self.available_timestamp <= decision_timestamp

    def get_if_available(self, decision_timestamp: datetime) -> Optional[T]:
        """The normal, safe way to consume a PointInTimeValue: returns the
        value if it was knowable by decision_timestamp, else None (never
        raises) -- matching this codebase's "unavailable, not fabricated"
        convention used throughout strategy/signal_engine.py."""
        return self.value if self.is_available_by(decision_timestamp) else None


def enforce_point_in_time(item: PointInTimeValue[T], decision_timestamp: datetime, label: str = "data item") -> T:
    """Returns item.value if available by decision_timestamp, else raises
    LookaheadViolationError. Use this at a call site where silently
    returning None would hide a bug (i.e. the caller already believes it
    pre-filtered and a violation here means that filtering failed)."""
    if not item.is_available_by(decision_timestamp):
        raise LookaheadViolationError(
            f"{label}: available_timestamp={item.available_timestamp} is AFTER "
            f"decision_timestamp={decision_timestamp} -- refusing to use it "
            f"(this would be look-ahead bias)."
        )
    return item.value


def filter_available(items: Iterable[PointInTimeValue], decision_timestamp: datetime) -> List[PointInTimeValue]:
    """Returns only the items that were actually available by
    decision_timestamp, preserving relative order. This is the primitive a
    real HistoricalNewsProvider/HistoricalSocialProvider implementation
    should use internally when it has a raw, unfiltered feed to narrow down
    for one `as_of` query."""
    return [i for i in items if i.is_available_by(decision_timestamp)]
