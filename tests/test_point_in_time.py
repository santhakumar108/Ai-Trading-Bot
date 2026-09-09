from datetime import datetime, timedelta, timezone

import pytest

from data.point_in_time import (
    LookaheadViolationError,
    PointInTimeValue,
    enforce_point_in_time,
    filter_available,
)


def dt(days_from_epoch: int) -> datetime:
    return datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=days_from_epoch)


def test_value_is_available_exactly_at_and_after_its_available_timestamp():
    item = PointInTimeValue(value=42, data_timestamp=dt(0), available_timestamp=dt(5))
    assert item.is_available_by(dt(5)) is True
    assert item.is_available_by(dt(6)) is True
    assert item.is_available_by(dt(4)) is False


def test_get_if_available_returns_none_not_a_fabricated_value_when_too_early():
    item = PointInTimeValue(value="quarterly EPS = 12.5", data_timestamp=dt(0), available_timestamp=dt(10))
    assert item.get_if_available(dt(9)) is None
    assert item.get_if_available(dt(10)) == "quarterly EPS = 12.5"


def test_available_timestamp_before_data_timestamp_is_rejected_as_a_bug():
    with pytest.raises(ValueError):
        PointInTimeValue(value=1, data_timestamp=dt(10), available_timestamp=dt(5))


def test_enforce_point_in_time_raises_on_a_lookahead_attempt():
    item = PointInTimeValue(value="news you shouldn't see yet", data_timestamp=dt(0), available_timestamp=dt(10))
    with pytest.raises(LookaheadViolationError):
        enforce_point_in_time(item, decision_timestamp=dt(9))
    # But succeeds once the decision timestamp has caught up.
    assert enforce_point_in_time(item, decision_timestamp=dt(10)) == "news you shouldn't see yet"


def test_filter_available_excludes_future_items_from_a_mixed_batch():
    items = [
        PointInTimeValue(value="old", data_timestamp=dt(0), available_timestamp=dt(1)),
        PointInTimeValue(value="borderline", data_timestamp=dt(4), available_timestamp=dt(5)),
        PointInTimeValue(value="future", data_timestamp=dt(9), available_timestamp=dt(10)),
    ]
    visible = filter_available(items, decision_timestamp=dt(5))
    values = [i.value for i in visible]
    assert values == ["old", "borderline"]
    assert "future" not in values


# --- Applied to the domain this spec cares most about: quarterly results --

def test_a_quarterly_result_released_after_a_historical_trade_date_is_not_available():
    """Direct proof of the spec's own example: a Q1 result for the quarter
    ending 2024-06-30, actually filed/announced on 2024-07-25, must not be
    visible to a decision made on 2024-07-10 -- even though the quarter
    itself (data_timestamp) is already in the past by then."""
    quarter_end = datetime(2024, 6, 30, tzinfo=timezone.utc)
    filed_date = datetime(2024, 7, 25, tzinfo=timezone.utc)
    result = PointInTimeValue(value={"eps": 12.5}, data_timestamp=quarter_end, available_timestamp=filed_date)

    decision_before_filing = datetime(2024, 7, 10, tzinfo=timezone.utc)
    decision_after_filing = datetime(2024, 7, 26, tzinfo=timezone.utc)

    assert result.get_if_available(decision_before_filing) is None
    assert result.get_if_available(decision_after_filing) == {"eps": 12.5}
