import pandas as pd
import pytest

from data.market_data import BatchFetchResult, DataUnavailableError, MarketDataProvider


def make_fetch_fn(df: pd.DataFrame):
    def fetch(symbol, period, interval):
        return df.copy()
    return fetch


def test_get_daily_uses_injected_fetch_fn(uptrend_daily):
    provider = MarketDataProvider(fetch_fn=make_fetch_fn(uptrend_daily))
    out = provider.get_daily("FAKE")
    assert len(out) == len(uptrend_daily)


def test_get_daily_is_cached(uptrend_daily):
    calls = {"n": 0}

    def fetch(symbol, period, interval):
        calls["n"] += 1
        return uptrend_daily.copy()

    provider = MarketDataProvider(fetch_fn=fetch)
    provider.get_daily("FAKE")
    provider.get_daily("FAKE")
    assert calls["n"] == 1


def test_average_volume_and_volatility(uptrend_daily):
    provider = MarketDataProvider(fetch_fn=make_fetch_fn(uptrend_daily))
    daily = provider.get_daily("FAKE")
    avg_vol = provider.average_volume(daily)
    hv = provider.historical_volatility(daily)
    assert avg_vol > 0
    assert hv > 0


def test_market_trend_classification(uptrend_daily, downtrend_daily):
    provider = MarketDataProvider()
    assert provider.market_trend(uptrend_daily) == "UPTREND"
    assert provider.market_trend(downtrend_daily) == "DOWNTREND"


def test_relative_strength_direction(uptrend_daily, downtrend_daily):
    provider = MarketDataProvider()
    rs = provider.relative_strength(uptrend_daily, downtrend_daily, window=20)
    assert rs > 0  # uptrend symbol vs downtrend "index" -> positive relative strength


def test_missing_columns_raise_data_unavailable():
    def fetch(symbol, period, interval):
        return pd.DataFrame({"Close": [1, 2, 3]})  # missing Open/High/Low/Volume

    provider = MarketDataProvider(fetch_fn=fetch)
    with pytest.raises(DataUnavailableError):
        provider.get_daily("FAKE")


def test_support_resistance_bounds(uptrend_daily):
    provider = MarketDataProvider()
    sr = provider.support_resistance(uptrend_daily, window=20)
    assert sr["support"] <= sr["resistance"]


# --- Hardening: retries, dedup, tz normalization, quality, corp actions ----

def test_transient_fetch_errors_are_retried_then_succeed(uptrend_daily, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)  # don't actually wait in tests
    calls = {"n": 0}

    def flaky_fetch(symbol, period, interval):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("simulated transient network failure")
        return uptrend_daily.copy()

    provider = MarketDataProvider(fetch_fn=flaky_fetch, max_retries=5, retry_backoff_seconds=0)
    out = provider.get_daily("FAKE")
    assert calls["n"] == 3
    assert len(out) == len(uptrend_daily)


def test_data_unavailable_error_is_never_retried(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    calls = {"n": 0}

    def always_unavailable(symbol, period, interval):
        calls["n"] += 1
        raise DataUnavailableError("no such symbol")

    provider = MarketDataProvider(fetch_fn=always_unavailable, max_retries=5, retry_backoff_seconds=0)
    with pytest.raises(DataUnavailableError):
        provider.get_daily("FAKE")
    assert calls["n"] == 1  # not retried -- "no data" is terminal, not transient


def test_persistent_transient_failure_eventually_raises(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)

    def always_flaky(symbol, period, interval):
        raise TimeoutError("simulated persistent timeout")

    provider = MarketDataProvider(fetch_fn=always_flaky, max_retries=3, retry_backoff_seconds=0)
    with pytest.raises(DataUnavailableError):
        provider.get_daily("FAKE")


# --- get_daily_batch: rate-limit-paced multi-symbol fetch -------------------

def test_get_daily_batch_no_sleep_when_batch_size_covers_all_symbols(uptrend_daily, monkeypatch):
    def fail_if_called(*_a, **_k):
        raise AssertionError("must not sleep when everything fits in one batch")
    monkeypatch.setattr("time.sleep", fail_if_called)

    provider = MarketDataProvider(fetch_fn=make_fetch_fn(uptrend_daily))
    symbols = ["A.NS", "B.NS", "C.NS"]
    result = provider.get_daily_batch(symbols, period="1y", batch_size=10, delay_between_batches_seconds=1.0)
    assert isinstance(result, BatchFetchResult)
    assert list(result.data.keys()) == symbols
    assert result.errors == {}


def test_get_daily_batch_sleeps_between_batches_not_after_last(uptrend_daily, monkeypatch):
    sleep_calls = []
    monkeypatch.setattr("time.sleep", lambda s: sleep_calls.append(s))

    provider = MarketDataProvider(fetch_fn=make_fetch_fn(uptrend_daily))
    symbols = ["A.NS", "B.NS", "C.NS", "D.NS", "E.NS"]
    result = provider.get_daily_batch(symbols, period="1y", batch_size=2, delay_between_batches_seconds=1.5)
    assert list(result.data.keys()) == symbols
    assert sleep_calls == [1.5, 1.5]  # batches of (2, 2, 1) -> 2 gaps, none after the last


def test_get_daily_batch_zero_delay_never_sleeps(uptrend_daily, monkeypatch):
    def fail_if_called(*_a, **_k):
        raise AssertionError("must not sleep with delay=0.0")
    monkeypatch.setattr("time.sleep", fail_if_called)

    provider = MarketDataProvider(fetch_fn=make_fetch_fn(uptrend_daily))
    result = provider.get_daily_batch(["A.NS", "B.NS", "C.NS"], period="1y", batch_size=1, delay_between_batches_seconds=0.0)
    assert len(result.data) == 3


def test_get_daily_batch_none_batch_size_means_one_batch_no_delay(uptrend_daily, monkeypatch):
    """batch_size/delay left as None (this provider holds no Config) means
    no batching at all -- every symbol fetched in one pass, no sleep."""
    def fail_if_called(*_a, **_k):
        raise AssertionError("must not sleep when batch_size/delay are None")
    monkeypatch.setattr("time.sleep", fail_if_called)

    provider = MarketDataProvider(fetch_fn=make_fetch_fn(uptrend_daily))
    result = provider.get_daily_batch(["A.NS", "B.NS", "C.NS"], period="1y")
    assert len(result.data) == 3


def test_get_daily_batch_fault_isolates_failed_symbol(uptrend_daily, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)

    def fetch(symbol, period, interval):
        if symbol == "BAD.NS":
            raise DataUnavailableError("simulated outage")
        return uptrend_daily.copy()

    provider = MarketDataProvider(fetch_fn=fetch)
    result = provider.get_daily_batch(
        ["GOOD1.NS", "BAD.NS", "GOOD2.NS"], period="1y", batch_size=1, delay_between_batches_seconds=0.0,
    )
    assert set(result.data.keys()) == {"GOOD1.NS", "GOOD2.NS"}
    assert "BAD.NS" in result.errors and "outage" in result.errors["BAD.NS"]


def test_duplicate_timestamps_are_dropped_keeping_the_last(uptrend_daily):
    dup = pd.concat([uptrend_daily.iloc[:5], uptrend_daily.iloc[4:5], uptrend_daily.iloc[5:]])
    provider = MarketDataProvider(fetch_fn=make_fetch_fn(dup))
    out = provider.get_daily("FAKE")
    assert out.index.duplicated().sum() == 0
    assert len(out) == len(uptrend_daily)


def test_naive_timestamps_are_tagged_as_kolkata_not_shifted(uptrend_daily):
    provider = MarketDataProvider(fetch_fn=make_fetch_fn(uptrend_daily))
    out = provider.get_daily("FAKE")
    assert out.index.tz is not None
    assert str(out.index.tz) in ("Asia/Kolkata", "+05:30")


def test_get_daily_with_quality_returns_ok_for_clean_data(uptrend_daily):
    provider = MarketDataProvider(fetch_fn=make_fetch_fn(uptrend_daily))
    as_of = uptrend_daily.index[-1] + pd.Timedelta(days=1)
    daily, report = provider.get_daily_with_quality("FAKE", as_of=as_of)
    assert daily is not None
    assert report.status in ("OK", "DEGRADED")  # synthetic fixture may trip the missing-rows/holiday checks
    assert report.symbol == "FAKE"


def test_get_daily_with_quality_never_raises_on_total_failure():
    def always_unavailable(symbol, period, interval):
        raise DataUnavailableError("no data")

    provider = MarketDataProvider(fetch_fn=always_unavailable, max_retries=1)
    daily, report = provider.get_daily_with_quality("FAKE")
    assert daily is None
    assert report.status == "INVALID"
    assert report.has_critical_issues is True


def test_get_corporate_actions_defaults_to_empty_not_fabricated():
    def failing_ca(symbol):
        raise RuntimeError("vendor down")

    provider = MarketDataProvider(corporate_actions_fetch_fn=failing_ca)
    ca = provider.get_corporate_actions("FAKE")
    assert ca.empty
    assert list(ca.columns) == ["date", "type", "value"]


def test_get_corporate_actions_returns_the_injected_real_history():
    import pandas as _pd

    def fake_ca(symbol):
        return _pd.DataFrame([{"date": "2024-01-15", "type": "split", "value": 2.0}])

    provider = MarketDataProvider(corporate_actions_fetch_fn=fake_ca)
    ca = provider.get_corporate_actions("FAKE")
    assert len(ca) == 1
    assert ca.iloc[0]["type"] == "split"
