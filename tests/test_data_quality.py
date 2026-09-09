from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from data.data_quality import DataQualityChecker
from data.market_calendar import get_calendar


def make_clean_daily(n=100, start="2024-01-02"):
    dates = pd.bdate_range(start=start, periods=n, tz="Asia/Kolkata")
    close = 100 + np.cumsum(np.random.default_rng(1).normal(0, 0.5, n))
    df = pd.DataFrame({
        "Open": close, "High": close + 1, "Low": close - 1, "Close": close,
        "Volume": np.full(n, 500_000.0),
    }, index=dates)
    return df


def test_clean_data_scores_high_and_status_ok_or_degraded():
    daily = make_clean_daily()
    checker = DataQualityChecker(calendar=get_calendar("generic"))  # weekend-only, matches bdate_range exactly
    report = checker.check("TEST", daily, as_of=daily.index[-1] + pd.Timedelta(days=1))
    assert report.quality_score > 0.9
    assert report.status in ("OK", "DEGRADED")
    assert report.has_critical_issues is False


def test_empty_dataframe_is_invalid():
    checker = DataQualityChecker()
    report = checker.check("TEST", pd.DataFrame())
    assert report.status == "INVALID"
    assert report.quality_score == 0.0


def test_none_is_invalid():
    checker = DataQualityChecker()
    report = checker.check("TEST", None)
    assert report.status == "INVALID"


def test_duplicate_timestamps_flagged_critical():
    daily = make_clean_daily()
    dup = pd.concat([daily.iloc[:3], daily.iloc[2:3], daily.iloc[3:]])
    checker = DataQualityChecker(calendar=get_calendar("generic"))
    report = checker.check("TEST", dup, as_of=dup.index[-1] + pd.Timedelta(days=1))
    assert any(i.check == "duplicate_timestamps" and i.severity == "CRITICAL" for i in report.issues)
    assert report.status == "INVALID"


def test_invalid_ohlc_flagged_critical():
    daily = make_clean_daily()
    daily = daily.copy()
    daily.iloc[5, daily.columns.get_loc("High")] = daily.iloc[5]["Low"] - 1  # High < Low
    checker = DataQualityChecker(calendar=get_calendar("generic"))
    report = checker.check("TEST", daily, as_of=daily.index[-1] + pd.Timedelta(days=1))
    assert any(i.check == "invalid_ohlc" for i in report.issues)
    assert report.status == "INVALID"


def test_negative_price_flagged_critical():
    daily = make_clean_daily()
    daily = daily.copy()
    daily.iloc[10, daily.columns.get_loc("Close")] = -5.0
    checker = DataQualityChecker(calendar=get_calendar("generic"))
    report = checker.check("TEST", daily)
    assert any(i.check == "invalid_ohlc" for i in report.issues)


def test_zero_volume_flagged():
    daily = make_clean_daily()
    daily = daily.copy()
    daily.iloc[0:15, daily.columns.get_loc("Volume")] = 0.0
    checker = DataQualityChecker(calendar=get_calendar("generic"))
    report = checker.check("TEST", daily, as_of=daily.index[-1] + pd.Timedelta(days=1))
    assert any(i.check == "zero_volume" for i in report.issues)


def test_abnormal_price_jump_unexplained_is_flagged_but_explained_jump_is_not():
    daily = make_clean_daily(n=60)
    daily = daily.copy()
    jump_idx = 30
    # A PERMANENT level shift from jump_idx onward (like an un-adjusted
    # split/bonus) -- only the jump day itself shows an abnormal return;
    # later days are all consistently scaled, so they don't also register
    # as jumps the way a one-day-only spike-and-revert would.
    for col in ("Open", "High", "Low", "Close"):
        daily.iloc[jump_idx:, daily.columns.get_loc(col)] *= 1.5
    checker = DataQualityChecker(calendar=get_calendar("generic"), abnormal_daily_return_threshold=0.20)

    unexplained_report = checker.check("TEST", daily, as_of=daily.index[-1] + pd.Timedelta(days=1))
    assert any(i.check == "abnormal_price_jump" for i in unexplained_report.issues)

    jump_date = daily.index[jump_idx].date()
    explained_report = checker.check(
        "TEST", daily, as_of=daily.index[-1] + pd.Timedelta(days=1),
        known_corporate_action_dates={jump_date},
    )
    assert not any(i.check == "abnormal_price_jump" for i in explained_report.issues)


def test_missing_rows_vs_calendar_flagged():
    daily = make_clean_daily(n=40)
    gappy = daily.drop(daily.index[10:18])  # remove 8 consecutive trading days
    checker = DataQualityChecker(calendar=get_calendar("generic"), max_missing_row_fraction=0.05)
    report = checker.check("TEST", gappy, as_of=gappy.index[-1] + pd.Timedelta(days=1))
    assert any(i.check == "missing_rows" for i in report.issues)


def test_stale_data_flagged_when_as_of_far_after_last_bar():
    daily = make_clean_daily(n=30)
    checker = DataQualityChecker(calendar=get_calendar("generic"), max_stale_data_days=5)
    far_future = daily.index[-1] + pd.Timedelta(days=30)
    report = checker.check("TEST", daily, as_of=far_future)
    assert any(i.check == "stale_data" for i in report.issues)


def test_is_tradeable_gate_reflects_threshold_and_critical_issues():
    daily = make_clean_daily()
    checker = DataQualityChecker(calendar=get_calendar("generic"), min_quality_score_to_trade=0.70)
    good_report = checker.check("TEST", daily, as_of=daily.index[-1] + pd.Timedelta(days=1))
    assert good_report.is_tradeable() is True

    dup = pd.concat([daily.iloc[:3], daily.iloc[2:3], daily.iloc[3:]])
    bad_report = checker.check("TEST", dup, as_of=dup.index[-1] + pd.Timedelta(days=1))
    assert bad_report.is_tradeable() is False
