"""
Market Data Engine
==================
Responsible for pulling OHLCV data (daily + intraday), volume, and deriving
simple market/sector/relative-strength context. Uses yfinance by default
(no API key required) behind a small provider interface so the data source
can be swapped later (broker feed, paid vendor) without touching any
downstream module.

This module does NOT compute trading signals -- see indicators/technical.py
and strategy/signal_engine.py for that. It only fetches and lightly shapes
data.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from data.data_quality import DataQualityChecker, DataQualityReport
from data.market_calendar import kolkata_tz

logger = logging.getLogger(__name__)

REQUIRED_OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


class DataUnavailableError(RuntimeError):
    """Raised when market data cannot be retrieved or is unusable. This is
    treated as a TERMINAL outcome for one fetch attempt (no data exists, or
    what came back is structurally unusable) -- see `_fetch_with_retries`,
    which retries only on OTHER exceptions (network/timeout errors), never
    on this one, since retrying "there is no such data" wastes time and
    never fabricates a result to fall back to."""


@dataclass
class MarketSnapshot:
    """A lightweight, point-in-time bundle of everything the signal engine
    needs about one symbol's data quality and context."""

    symbol: str
    last_price: float
    daily: pd.DataFrame              # OHLCV, daily bars, ascending by date
    intraday: Optional[pd.DataFrame]  # OHLCV, intraday bars (may be None)
    avg_volume_20d: float
    index_daily: Optional[pd.DataFrame]
    as_of: datetime

    @property
    def has_sufficient_history(self) -> bool:
        return len(self.daily) >= 60  # signal engine enforces its own stricter min


@dataclass
class BatchFetchResult:
    """Result of `MarketDataProvider.get_daily_batch`. `errors` holds one
    entry per symbol whose fetch raised `DataUnavailableError` -- callers
    decide how/whether to log each failure; this dataclass only carries
    data, matching `get_daily`'s own "raise, don't log" contract."""
    data: Dict[str, pd.DataFrame] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)


def _default_fetch_history(
    symbol: str, period: str, interval: str, timeout: float = 15.0
) -> pd.DataFrame:
    """Default fetch function backed by yfinance. Isolated in its own
    function so tests can monkeypatch MarketDataProvider.fetch_fn instead
    of hitting the network. `timeout` is passed straight through to
    yfinance's own HTTP client -- a hung request fails instead of blocking
    forever, and surfaces as a retryable exception to `_fetch_with_retries`."""
    import yfinance as yf

    df = yf.download(
        symbol,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=False,
        multi_level_index=False,
        timeout=timeout,
    )
    if df is None or df.empty:
        raise DataUnavailableError(f"No data returned for {symbol} ({period}/{interval})")
    df = df.rename(columns=str.title)
    missing = [c for c in REQUIRED_OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise DataUnavailableError(f"{symbol}: missing columns {missing}")
    return df[REQUIRED_OHLCV_COLUMNS].copy()


def _default_fetch_corporate_actions(symbol: str) -> pd.DataFrame:
    """Default corporate-action fetch backed by yfinance's real
    `Ticker.splits` / `Ticker.dividends` history -- actual historical
    record, not fabricated. Returns an empty frame (never a guess) if the
    vendor has nothing or the request fails."""
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    rows = []
    try:
        for ts, ratio in ticker.splits.items():
            rows.append({"date": ts.date() if hasattr(ts, "date") else ts, "type": "split", "value": float(ratio)})
    except Exception as exc:
        logger.warning("Could not fetch split history for %s: %s", symbol, exc)
    try:
        for ts, amount in ticker.dividends.items():
            rows.append({"date": ts.date() if hasattr(ts, "date") else ts, "type": "dividend", "value": float(amount)})
    except Exception as exc:
        logger.warning("Could not fetch dividend history for %s: %s", symbol, exc)
    return pd.DataFrame(rows, columns=["date", "type", "value"])


def _fetch_with_retries(
    fetch_fn: Callable[[str, str, str], pd.DataFrame],
    symbol: str, period: str, interval: str,
    max_retries: int = 3, backoff_seconds: float = 1.0,
) -> pd.DataFrame:
    """Retries transient failures (network errors, timeouts) with
    exponential backoff. `DataUnavailableError` is NOT retried -- it means
    "this data genuinely doesn't exist / is structurally unusable," and
    retrying that wastes time and risks masking a real problem behind a
    delay. Any other exception is assumed transient."""
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            return fetch_fn(symbol, period, interval)
        except DataUnavailableError:
            raise
        except Exception as exc:  # network/timeout/vendor hiccup
            last_exc = exc
            logger.warning(
                "Fetch attempt %d/%d for %s (%s/%s) failed: %s",
                attempt + 1, max_retries, symbol, period, interval, exc,
            )
            if attempt < max_retries - 1:
                time.sleep(backoff_seconds * (2 ** attempt))
    raise DataUnavailableError(f"{symbol}: fetch failed after {max_retries} attempts: {last_exc}")


def classify_regime(
    daily: pd.DataFrame, i: Optional[int] = None, trend_window: int = 50, vol_window: int = 20,
) -> str:
    """
    Transparent, trailing-only market-regime classifier: `"{BULL|BEAR|
    SIDEWAYS}_{HIGH_VOL|LOW_VOL}"` (or `"UNKNOWN"` with insufficient
    history). Originally lived in `backtesting/backtester.py` (used only
    for post-hoc regime segmentation of backtest results) -- relocated
    here, unchanged, so the SAME classifier can also run during live/paper
    scanning (spec Part 5.6 "market regime" alpha family), matching the
    pattern already established for `market_trend`/`relative_strength`.

    `i`: the bar index to classify AS OF (uses only `daily.iloc[:i+1]` --
    never looks past it, so a backtest calling this per-bar stays
    point-in-time-safe). `None` (the live-scan case) means "classify as of
    the LAST available bar" -- equivalent to `i = len(daily) - 1`.
    """
    if i is None:
        i = len(daily) - 1
    if i < max(trend_window, vol_window) + 1:
        return "UNKNOWN"
    window = daily.iloc[max(0, i - trend_window): i + 1]
    ret = window["Close"].iloc[-1] / window["Close"].iloc[0] - 1
    vol_window_slice = daily["Close"].iloc[max(0, i - vol_window): i + 1]
    log_ret = np.log(vol_window_slice / vol_window_slice.shift(1)).dropna()
    ann_vol = log_ret.std() * np.sqrt(252) if len(log_ret) > 2 else np.nan

    trend = "BULL" if ret > 0.05 else ("BEAR" if ret < -0.05 else "SIDEWAYS")
    if np.isnan(ann_vol):
        vol_label = "UNKNOWN_VOL"
    else:
        vol_label = "HIGH_VOL" if ann_vol > 0.35 else "LOW_VOL"
    return f"{trend}_{vol_label}"


class MarketDataProvider:
    """
    Fetches and lightly validates market data.

    Parameters
    ----------
    fetch_fn:
        Callable(symbol, period, interval) -> DataFrame[Open,High,Low,Close,Volume].
        Defaults to a yfinance-backed implementation. Inject a fake for tests
        or to swap data vendors.
    corporate_actions_fetch_fn:
        Callable(symbol) -> DataFrame[date,type,value]. Defaults to a
        yfinance-backed implementation (real split/dividend history).
    max_retries / retry_backoff_seconds:
        Retry policy for transient fetch failures (network errors, timeouts)
        -- see `_fetch_with_retries`. Does not retry a `DataUnavailableError`
        (data that genuinely doesn't exist).
    quality_checker:
        A `data.data_quality.DataQualityChecker` used by `get_daily_with_quality`.
        Defaults to one built from `data.market_calendar.get_calendar("NSE")`.
    """

    def __init__(
        self,
        fetch_fn: Optional[Callable[[str, str, str], pd.DataFrame]] = None,
        corporate_actions_fetch_fn: Optional[Callable[[str], pd.DataFrame]] = None,
        max_retries: int = 3,
        retry_backoff_seconds: float = 1.0,
        quality_checker: Optional[DataQualityChecker] = None,
    ):
        self.fetch_fn = fetch_fn or _default_fetch_history
        self.corporate_actions_fetch_fn = corporate_actions_fetch_fn or _default_fetch_corporate_actions
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.quality_checker = quality_checker or DataQualityChecker()
        self._cache: Dict[str, pd.DataFrame] = {}
        self._ca_cache: Dict[str, pd.DataFrame] = {}

    @staticmethod
    def _validate(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Enforced regardless of which fetch_fn produced the data (default
        yfinance-backed one, or an injected/custom source) -- callers
        downstream assume these columns exist, sorted ascending, no
        duplicate timestamps, and a timezone-aware (Asia/Kolkata) index.

        This is data HYGIENE, not fabrication: dropping an exact duplicate
        row or attaching a timezone label to an already-correct wall-clock
        timestamp changes no price/volume value. Genuine quality problems
        (invalid OHLC, missing rows, stale data, ...) are left in place and
        surfaced via `DataQualityChecker` instead -- this method never
        repairs or invents a price."""
        if df is None or df.empty:
            raise DataUnavailableError(f"No data returned for {symbol}")
        missing = [c for c in REQUIRED_OHLCV_COLUMNS if c not in df.columns]
        if missing:
            raise DataUnavailableError(f"{symbol}: missing columns {missing}")
        df = df[REQUIRED_OHLCV_COLUMNS].copy()

        if df.index.duplicated().any():
            dup_n = int(df.index.duplicated().sum())
            logger.warning("%s: dropping %d duplicate timestamp row(s) (keeping the last).", symbol, dup_n)
            df = df[~df.index.duplicated(keep="last")]

        df = df.sort_index()

        if getattr(df.index, "tz", None) is None:
            tz = kolkata_tz()
            if tz is not None:
                try:
                    df.index = df.index.tz_localize(tz)
                except TypeError:
                    pass  # already tz-aware in a way pandas didn't report cleanly; leave as-is

        return df

    # ------------------------------------------------------------------
    # Raw fetchers
    # ------------------------------------------------------------------
    def get_daily(self, symbol: str, period: str = "2y") -> pd.DataFrame:
        cache_key = f"{symbol}:daily:{period}"
        if cache_key not in self._cache:
            raw = _fetch_with_retries(
                self.fetch_fn, symbol, period, "1d",
                max_retries=self.max_retries, backoff_seconds=self.retry_backoff_seconds,
            )
            self._cache[cache_key] = self._validate(raw, symbol)
        return self._cache[cache_key].copy()

    def get_daily_batch(
        self,
        symbols: List[str],
        period: str = "2y",
        batch_size: Optional[int] = None,
        delay_between_batches_seconds: Optional[float] = None,
    ) -> "BatchFetchResult":
        """
        Fetches `get_daily(symbol, period=period)` for every symbol in
        `symbols`, in batches of `batch_size` with a pause of
        `delay_between_batches_seconds` between batches (never after the
        last one) -- the same batching shape `paper_trading/scanner.py`'s
        `UniverseScanner.scan()` already applies to the live-scan path,
        reusable here for `run_universe_backtest` (spec Part 22) and
        `run_small_account_survival` (spec Part 25) so a large universe
        doesn't fire dozens of back-to-back requests at a rate-limited
        free data API.

        `batch_size`/`delay_between_batches_seconds` left as `None` means
        "no batching, no delay" -- this provider holds no `Config`
        reference by design; callers wanting the project's configured
        pacing pass `config.universe.scan_batch_size`/
        `scan_batch_delay_seconds` explicitly.

        Fault-isolated per symbol: a `DataUnavailableError` fetching one
        symbol is caught, recorded in the returned `errors` dict
        (symbol -> str(exc)), and does not abort the remaining batch. Any
        other exception is not caught here, matching `get_daily`'s own
        contract. `data` preserves the same order as `symbols`, containing
        only symbols that fetched successfully.
        """
        n = len(symbols)
        size = max(1, batch_size) if batch_size is not None else max(1, n)
        delay = max(0.0, delay_between_batches_seconds) if delay_between_batches_seconds is not None else 0.0

        result = BatchFetchResult()
        for batch_start in range(0, n, size):
            batch = symbols[batch_start: batch_start + size]
            for symbol in batch:
                try:
                    result.data[symbol] = self.get_daily(symbol, period=period)
                except DataUnavailableError as exc:
                    result.errors[symbol] = str(exc)

            is_last_batch = batch_start + size >= n
            if not is_last_batch and delay > 0:
                time.sleep(delay)

        return result

    def get_daily_with_quality(
        self, symbol: str, period: str = "2y", as_of: Optional[datetime] = None,
    ) -> "tuple[Optional[pd.DataFrame], DataQualityReport]":
        """Never raises: if the fetch fails outright, returns
        (None, DataQualityReport(status='INVALID', ...)) instead of
        propagating `DataUnavailableError`, so callers that want an
        explicit data-quality status rather than an exception (spec
        section 2: "if a provider is unavailable, return an explicit
        data-quality status") can get one directly."""
        try:
            daily = self.get_daily(symbol, period=period)
        except DataUnavailableError as exc:
            from data.data_quality import DataQualityIssue, DataQualityReport as _Report
            report = _Report(
                symbol=symbol, as_of=as_of or datetime.now(timezone.utc),
                issues=[DataQualityIssue("provider_unavailable", "CRITICAL", str(exc))],
                quality_score=0.0, status="INVALID",
                min_quality_score_to_trade=self.quality_checker.min_quality_score_to_trade,
            )
            return None, report

        known_ca_dates = set()
        try:
            ca = self.get_corporate_actions(symbol)
            known_ca_dates = set(ca["date"]) if not ca.empty else set()
        except Exception:
            pass  # corporate-action awareness is best-effort; absence isn't a fetch failure

        report = self.quality_checker.check(symbol, daily, as_of=as_of, known_corporate_action_dates=known_ca_dates)
        return daily, report

    def get_corporate_actions(self, symbol: str) -> pd.DataFrame:
        """Real historical split/dividend record (see
        `_default_fetch_corporate_actions`), never fabricated. Returns an
        empty DataFrame (not None) if unavailable, so callers can always
        treat the result uniformly."""
        if symbol not in self._ca_cache:
            try:
                self._ca_cache[symbol] = self.corporate_actions_fetch_fn(symbol)
            except Exception as exc:
                logger.warning("Corporate actions unavailable for %s: %s", symbol, exc)
                self._ca_cache[symbol] = pd.DataFrame(columns=["date", "type", "value"])
        return self._ca_cache[symbol].copy()

    def get_intraday(self, symbol: str, period: str = "5d", interval: str = "15m") -> Optional[pd.DataFrame]:
        cache_key = f"{symbol}:intraday:{period}:{interval}"
        if cache_key in self._cache:
            return self._cache[cache_key].copy()
        try:
            raw = _fetch_with_retries(
                self.fetch_fn, symbol, period, interval,
                max_retries=self.max_retries, backoff_seconds=self.retry_backoff_seconds,
            )
            df = self._validate(raw, symbol)
            self._cache[cache_key] = df
            return df.copy()
        except DataUnavailableError:
            logger.warning("Intraday data unavailable for %s; continuing with daily only.", symbol)
            return None

    def get_quote(self, symbol: str) -> float:
        """Last traded price, derived from the most recent daily bar's close
        (or the last intraday bar if available). For paper trading this
        stands in for a real-time quote feed."""
        intraday = self.get_intraday(symbol)
        if intraday is not None and not intraday.empty:
            return float(intraday["Close"].iloc[-1])
        daily = self.get_daily(symbol, period="5d")
        return float(daily["Close"].iloc[-1])

    # ------------------------------------------------------------------
    # Derived context
    # ------------------------------------------------------------------
    def average_volume(self, daily: pd.DataFrame, window: int = 20) -> float:
        if len(daily) < 1:
            return 0.0
        return float(daily["Volume"].tail(window).mean())

    def historical_volatility(self, daily: pd.DataFrame, window: int = 20) -> float:
        """Annualized close-to-close volatility (std of log returns * sqrt(252))."""
        if len(daily) < window + 1:
            return float("nan")
        log_ret = np.log(daily["Close"] / daily["Close"].shift(1)).dropna()
        return float(log_ret.tail(window).std() * np.sqrt(252))

    def relative_strength(self, symbol_daily: pd.DataFrame, index_daily: pd.DataFrame, window: int = 20) -> float:
        """Simple relative strength: symbol return minus index return over
        `window` bars. Positive = outperforming the benchmark."""
        if len(symbol_daily) < window + 1 or len(index_daily) < window + 1:
            return float("nan")
        sym_ret = symbol_daily["Close"].iloc[-1] / symbol_daily["Close"].iloc[-window] - 1
        idx_ret = index_daily["Close"].iloc[-1] / index_daily["Close"].iloc[-window] - 1
        return float(sym_ret - idx_ret)

    def support_resistance(self, daily: pd.DataFrame, window: int = 20) -> Dict[str, float]:
        """Naive but transparent support/resistance: rolling window low/high
        plus the most recent swing points. Good enough as one input among
        many -- not a precision tool."""
        if len(daily) < window:
            window = len(daily)
        recent = daily.tail(window)
        return {
            "support": float(recent["Low"].min()),
            "resistance": float(recent["High"].max()),
        }

    def market_trend(self, index_daily: pd.DataFrame, fast: int = 20, slow: int = 50) -> str:
        """Very simple trend classification of the benchmark index using a
        fast/slow moving-average cross. Used as one of several market-
        condition inputs, never as the sole basis for a decision."""
        if len(index_daily) < slow:
            return "UNKNOWN"
        fast_ma = index_daily["Close"].tail(fast).mean()
        slow_ma = index_daily["Close"].tail(slow).mean()
        if fast_ma > slow_ma * 1.01:
            return "UPTREND"
        if fast_ma < slow_ma * 0.99:
            return "DOWNTREND"
        return "SIDEWAYS"

    # ------------------------------------------------------------------
    # High-level snapshot used by the strategy layer
    # ------------------------------------------------------------------
    def build_snapshot(self, symbol: str, index_symbol: Optional[str] = None) -> MarketSnapshot:
        daily = self.get_daily(symbol)
        intraday = self.get_intraday(symbol)
        index_daily = self.get_daily(index_symbol) if index_symbol else None
        return MarketSnapshot(
            symbol=symbol,
            last_price=self.get_quote(symbol),
            daily=daily,
            intraday=intraday,
            avg_volume_20d=self.average_volume(daily),
            index_daily=index_daily,
            as_of=datetime.now(timezone.utc),
        )
