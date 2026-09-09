from datetime import date, datetime

from data.market_calendar import NSECalendar, get_calendar, localize_to_kolkata


def test_weekends_are_never_trading_days():
    cal = NSECalendar()
    saturday = date(2024, 6, 1)   # a known Saturday
    sunday = date(2024, 6, 2)
    assert saturday.weekday() == 5 and sunday.weekday() == 6
    assert cal.is_trading_day(saturday) is False
    assert cal.is_trading_day(sunday) is False


def test_fixed_national_holidays_are_always_closed_every_year():
    cal = NSECalendar()
    for year in (2023, 2024, 2025, 2030):
        assert cal.is_trading_day(date(year, 1, 26)) is False   # Republic Day
        assert cal.is_trading_day(date(year, 8, 15)) is False   # Independence Day
        assert cal.is_trading_day(date(year, 10, 2)) is False   # Gandhi Jayanti


def test_ordinary_weekday_is_a_trading_day():
    cal = NSECalendar()
    # A Tuesday that is not one of the fixed holidays.
    assert cal.is_trading_day(date(2024, 6, 4)) is True


def test_extra_holidays_are_respected_without_inventing_dates():
    extra = {date(2024, 11, 1)}  # e.g. a Diwali/Muhurat-style holiday the caller supplies
    cal = NSECalendar(extra_holidays=extra)
    assert cal.is_trading_day(date(2024, 11, 1)) is False
    assert cal.holiday_reason(date(2024, 11, 1)) == "configured NSE holiday"
    # A calendar with NO extra holidays supplied must not guess this date is closed.
    plain_cal = NSECalendar()
    assert plain_cal.is_trading_day(date(2024, 11, 1)) is True


def test_next_and_previous_trading_day_skip_weekends_and_holidays():
    cal = NSECalendar()
    friday_before_independence_day = date(2025, 8, 15)  # a Friday, fixed holiday
    nxt = cal.next_trading_day(friday_before_independence_day)
    assert cal.is_trading_day(nxt)
    assert nxt > friday_before_independence_day

    prev = cal.previous_trading_day(friday_before_independence_day)
    assert cal.is_trading_day(prev)
    assert prev < friday_before_independence_day


def test_missing_trading_days_flags_gaps_in_a_series():
    cal = NSECalendar()
    start, end = date(2024, 6, 3), date(2024, 6, 7)  # Mon-Fri, all trading days
    present = [date(2024, 6, 3), date(2024, 6, 4), date(2024, 6, 7)]  # missing Wed/Thu
    missing = cal.missing_trading_days(present, start, end)
    assert date(2024, 6, 5) in missing
    assert date(2024, 6, 6) in missing
    assert date(2024, 6, 3) not in missing


def test_get_calendar_factory_only_applies_fixed_holidays_for_nse():
    nse = get_calendar("NSE")
    assert nse.is_trading_day(date(2024, 8, 15)) is False

    generic = get_calendar("generic")
    assert generic.is_trading_day(date(2024, 8, 15)) is True   # honestly weekend-only
    assert generic.is_trading_day(date(2024, 6, 1)) is False   # still a Saturday


def test_from_csv_loads_extra_holidays(tmp_path):
    csv_path = tmp_path / "holidays.csv"
    csv_path.write_text("date\n2024-11-01\n2024-03-25\n")
    cal = NSECalendar.from_csv(str(csv_path))
    assert date(2024, 11, 1) in cal.extra_holidays
    assert date(2024, 3, 25) in cal.extra_holidays
    assert cal.is_trading_day(date(2024, 11, 1)) is False


def test_localize_to_kolkata_tags_naive_datetime_without_shifting_clock_time():
    naive = datetime(2024, 6, 3, 10, 0, 0)
    localized = localize_to_kolkata(naive)
    assert localized.hour == 10 and localized.minute == 0  # not shifted, just tagged
    assert localized.tzinfo is not None


def test_is_market_hours_respects_nse_session_not_us_hours():
    cal = NSECalendar()
    during_session = localize_to_kolkata(datetime(2024, 6, 4, 10, 0, 0))   # Tue, 10:00 IST
    before_open = localize_to_kolkata(datetime(2024, 6, 4, 8, 0, 0))       # Tue, 08:00 IST
    after_close = localize_to_kolkata(datetime(2024, 6, 4, 16, 0, 0))      # Tue, 16:00 IST
    assert cal.is_market_hours(during_session) is True
    assert cal.is_market_hours(before_open) is False
    assert cal.is_market_hours(after_close) is False
