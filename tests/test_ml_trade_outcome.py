"""
Spec Part 7: the ML trade-outcome meta-model (triple-barrier target-before-
stop labeling) and its live/backtest wiring.

Covers:
  * Triple-barrier labeling correctness (target-first, stop-first, neither-
    within-horizon-so-dropped, same-bar-tie stop priority).
  * Feature leakage-safety (unaffected by appending future bars).
  * `MLBaselineModel(feature_cols=...)` backward compatibility.
  * `HistoricalMLProvider.fit_on_training_window(label_mode="trade_outcome")`
    still honors the cutoff-date no-lookahead guard.
  * `LiveTradeOutcomeMLProvider` caching (fit once per symbol) and graceful
    `None` on insufficient history / unusable AUC.
  * `PaperTradingEngine` wiring: `ml_probability_up` reaches the report and
    decision log ONLY when an ml_provider is supplied; stays `None`
    (today's exact behavior) otherwise.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from config.settings import Config
from data.market_data import MarketDataProvider
from fundamentals.fundamental_analysis import FundamentalAnalyzer
from indicators.technical import atr as _atr
from models.ml_baseline import (
    DIRECTION_FEATURE_COLS, TRADE_OUTCOME_FEATURE_COLS, MLBaselineModel, build_trade_outcome_features,
)
from paper_trading.decision_log import DecisionLog
from paper_trading.engine import PaperTradingEngine
from paper_trading.journal import TradeJournal
from paper_trading.live_ml_provider import LiveTradeOutcomeMLProvider
from tests.conftest import make_choppy_ohlcv


# =============================================================================
# Triple-barrier labeling correctness
# =============================================================================

ATR_WINDOW = 14
STOP_MULT = 1.5
TARGET_MULT = 3.0


def _engineer_barrier_case(daily: pd.DataFrame, test_idx: int, outcome: str, hit_at: int = 3):
    """Starting from a real (feature-valid) synthetic series, forces bars
    `test_idx+1 .. test_idx+hit_at-1` to stay tame (no barrier touched),
    then forces bar `test_idx+hit_at` to realize the requested `outcome`.
    Returns (edited_df, stop, target) computed from the REAL ATR at
    test_idx so the engineered levels match what build_trade_outcome_features
    will itself compute."""
    df = daily.copy()
    high_col, low_col, close_col = df.columns.get_loc("High"), df.columns.get_loc("Low"), df.columns.get_loc("Close")
    entry = float(df["Close"].iloc[test_idx])
    atr_val = float(_atr(df["High"], df["Low"], df["Close"], ATR_WINDOW).iloc[test_idx])
    stop = entry - STOP_MULT * atr_val
    target = entry + TARGET_MULT * atr_val

    for j in range(test_idx + 1, test_idx + hit_at):
        df.iloc[j, high_col] = entry
        df.iloc[j, low_col] = entry
        df.iloc[j, close_col] = entry

    j = test_idx + hit_at
    if outcome == "target":
        df.iloc[j, high_col] = target + 1.0
        df.iloc[j, low_col] = entry
    elif outcome == "stop":
        df.iloc[j, low_col] = stop - 1.0
        df.iloc[j, high_col] = entry
    elif outcome == "tie":
        df.iloc[j, high_col] = target + 1.0
        df.iloc[j, low_col] = stop - 1.0
    elif outcome == "neither":
        for k in range(test_idx + 1, test_idx + 1 + 25):  # well past any horizon used in these tests
            df.iloc[k, high_col] = entry
            df.iloc[k, low_col] = entry
            df.iloc[k, close_col] = entry
    else:
        raise ValueError(outcome)
    return df, stop, target


@pytest.fixture
def base_daily():
    # 200 bars gives ample warmup (sma50/adx/etc.) with room after a mid-
    # series test bar for barrier resolution.
    return make_choppy_ohlcv(n=200, seed=5)


def test_target_hit_first_labels_one(base_daily):
    test_idx = 100
    df, stop, target = _engineer_barrier_case(base_daily, test_idx, "target", hit_at=3)
    feats = build_trade_outcome_features(
        df, atr_window=ATR_WINDOW, stop_atr_multiple=STOP_MULT, target_atr_multiple=TARGET_MULT, max_holding_days=10,
    )
    ts = df.index[test_idx]
    assert ts in feats.index
    assert feats.loc[ts, "label"] == pytest.approx(1.0)


def test_stop_hit_first_labels_zero(base_daily):
    test_idx = 100
    df, stop, target = _engineer_barrier_case(base_daily, test_idx, "stop", hit_at=3)
    feats = build_trade_outcome_features(
        df, atr_window=ATR_WINDOW, stop_atr_multiple=STOP_MULT, target_atr_multiple=TARGET_MULT, max_holding_days=10,
    )
    ts = df.index[test_idx]
    assert ts in feats.index
    assert feats.loc[ts, "label"] == pytest.approx(0.0)


def test_same_bar_tie_prioritizes_stop(base_daily):
    """Daily OHLC can't tell us the true intraday order -- the conservative
    convention (matching backtesting/backtester.py's exit logic) is: if a
    bar's range spans BOTH the stop and the target, treat it as a loss."""
    test_idx = 100
    df, stop, target = _engineer_barrier_case(base_daily, test_idx, "tie", hit_at=3)
    feats = build_trade_outcome_features(
        df, atr_window=ATR_WINDOW, stop_atr_multiple=STOP_MULT, target_atr_multiple=TARGET_MULT, max_holding_days=10,
    )
    ts = df.index[test_idx]
    assert ts in feats.index
    assert feats.loc[ts, "label"] == pytest.approx(0.0)


def test_neither_barrier_hit_within_horizon_is_dropped_not_fabricated(base_daily):
    test_idx = 100
    df, _, _ = _engineer_barrier_case(base_daily, test_idx, "neither")
    feats = build_trade_outcome_features(
        df, atr_window=ATR_WINDOW, stop_atr_multiple=STOP_MULT, target_atr_multiple=TARGET_MULT, max_holding_days=10,
    )
    ts = df.index[test_idx]
    assert ts not in feats.index  # no resolved outcome -> row dropped, never guessed


def test_labels_are_only_0_or_1():
    daily = make_choppy_ohlcv(n=300, seed=9)
    feats = build_trade_outcome_features(daily)
    assert set(feats["label"].unique()).issubset({0.0, 1.0})


def test_the_final_bar_can_never_be_labeled():
    """The very last row has zero forward bars to check a barrier against
    -- structurally unresolvable regardless of how the market behaves."""
    daily = make_choppy_ohlcv(n=200, seed=3)
    feats = build_trade_outcome_features(daily, max_holding_days=10)
    assert daily.index[-1] not in feats.index


# =============================================================================
# Feature leakage-safety
# =============================================================================

def test_features_at_a_row_unaffected_by_appending_future_bars():
    daily = make_choppy_ohlcv(n=250, seed=11)
    test_idx = 100
    # Force a DETERMINISTIC, early (bar test_idx+3) resolution so both the
    # full and truncated series resolve `ts` identically -- relying on the
    # base series' own natural volatility to resolve within the horizon is
    # not guaranteed (a quiet stretch can legitimately go unresolved).
    engineered, _, _ = _engineer_barrier_case(daily, test_idx, "target", hit_at=3)
    ts = engineered.index[test_idx]

    full = build_trade_outcome_features(engineered, max_holding_days=10)
    # Truncate well past test_idx's own resolution window (10 bars) but far
    # short of the full series -- if row `ts` survives dropna in both, its
    # FEATURE columns (not label, which is unaffected here too since the
    # resolving bars are all included) must be identical.
    truncated = build_trade_outcome_features(engineered.iloc[: test_idx + 30], max_holding_days=10)

    assert ts in full.index and ts in truncated.index
    for col in TRADE_OUTCOME_FEATURE_COLS:
        assert full.loc[ts, col] == pytest.approx(truncated.loc[ts, col], nan_ok=False)
    assert full.loc[ts, "label"] == pytest.approx(truncated.loc[ts, "label"])


# =============================================================================
# MLBaselineModel(feature_cols=...) backward compatibility
# =============================================================================

def test_default_feature_cols_unchanged():
    model = MLBaselineModel(n_splits=3)
    assert model.feature_cols == DIRECTION_FEATURE_COLS


def test_explicit_trade_outcome_feature_cols_accepted_and_fits():
    daily = make_choppy_ohlcv(n=300, seed=13)
    features = build_trade_outcome_features(daily)
    model = MLBaselineModel(n_splits=3, feature_cols=TRADE_OUTCOME_FEATURE_COLS)
    assert model.feature_cols == TRADE_OUTCOME_FEATURE_COLS
    perf = model.fit(features)
    assert 0.0 <= perf.auc <= 1.0
    prob = model.predict_proba_up(features.iloc[-1])
    assert prob is None or 0.0 <= prob <= 1.0


# =============================================================================
# HistoricalMLProvider: trade_outcome mode still honors the cutoff guard
# =============================================================================

def test_historical_ml_provider_trade_outcome_mode_respects_cutoff():
    from backtesting.historical_providers import HistoricalMLProvider

    daily = make_choppy_ohlcv(n=600, seed=17)
    train = daily.iloc[:400]
    provider = HistoricalMLProvider.fit_on_training_window(
        train_daily=train, full_daily=daily, label_mode="trade_outcome", min_auc=0.0,  # accept any AUC for this test
    )
    if provider is None:
        pytest.skip("Model not usable on this synthetic seed at min_auc=0.0 -- not the property under test.")

    cutoff = train.index[-1]
    at_cutoff = provider.get("TEST", cutoff)
    before_cutoff = provider.get("TEST", train.index[100])
    after_cutoff = provider.get("TEST", daily.index[450])
    assert at_cutoff is None
    assert before_cutoff is None
    assert after_cutoff is None or 0.0 <= after_cutoff <= 1.0


def test_historical_ml_provider_direction_mode_default_unchanged():
    """label_mode defaults to 'direction' -- omitting it must behave exactly
    as before this phase (same call path as pre-Phase-3 code)."""
    from backtesting.historical_providers import HistoricalMLProvider

    daily = make_choppy_ohlcv(n=300, seed=19)
    train = daily.iloc[:200]
    provider = HistoricalMLProvider.fit_on_training_window(train_daily=train, full_daily=daily)
    # No assertion on usability (depends on synthetic data) -- the point is
    # this call succeeds with the old signature/defaults, unchanged.
    assert provider is None or provider.cutoff_date == train.index[-1]


# =============================================================================
# LiveTradeOutcomeMLProvider: caching + graceful unavailability
# =============================================================================

def test_live_provider_fits_at_most_once_per_symbol(monkeypatch):
    daily = make_choppy_ohlcv(n=400, seed=23)
    provider = LiveTradeOutcomeMLProvider(min_history_bars=100, min_auc=0.0, n_splits=3)

    call_count = {"n": 0}
    original_fit = MLBaselineModel.fit

    def counting_fit(self, features):
        call_count["n"] += 1
        return original_fit(self, features)

    monkeypatch.setattr(MLBaselineModel, "fit", counting_fit)

    first = provider.predict("SYM", daily)
    second = provider.predict("SYM", daily)
    assert call_count["n"] == 1  # second call served from cache, no re-fit
    assert first == second or (first is None and second is None)


def test_live_provider_refresh_forces_a_new_fit(monkeypatch):
    daily = make_choppy_ohlcv(n=400, seed=29)
    provider = LiveTradeOutcomeMLProvider(min_history_bars=100, min_auc=0.0, n_splits=3)

    call_count = {"n": 0}
    original_fit = MLBaselineModel.fit

    def counting_fit(self, features):
        call_count["n"] += 1
        return original_fit(self, features)

    monkeypatch.setattr(MLBaselineModel, "fit", counting_fit)

    provider.predict("SYM", daily)
    provider.refresh("SYM", daily)
    assert call_count["n"] == 2


def test_live_provider_returns_none_on_insufficient_history():
    tiny = make_choppy_ohlcv(n=50, seed=31)
    provider = LiveTradeOutcomeMLProvider(min_history_bars=300)
    assert provider.predict("SYM", tiny) is None


def test_live_provider_returns_none_when_auc_gate_not_cleared():
    daily = make_choppy_ohlcv(n=400, seed=37)
    # An impossibly strict AUC gate -- no real model clears 0.999 out-of-sample.
    provider = LiveTradeOutcomeMLProvider(min_history_bars=100, min_auc=0.999, n_splits=3)
    assert provider.predict("SYM", daily) is None


def test_live_provider_caches_unusable_result_without_refitting(monkeypatch):
    daily = make_choppy_ohlcv(n=400, seed=41)
    provider = LiveTradeOutcomeMLProvider(min_history_bars=100, min_auc=0.999, n_splits=3)

    call_count = {"n": 0}
    original_fit = MLBaselineModel.fit

    def counting_fit(self, features):
        call_count["n"] += 1
        return original_fit(self, features)

    monkeypatch.setattr(MLBaselineModel, "fit", counting_fit)

    provider.predict("SYM", daily)
    provider.predict("SYM", daily)
    assert call_count["n"] == 1  # "not usable" cached too -- not retried every call


# =============================================================================
# PaperTradingEngine wiring
# =============================================================================

def make_offline_engine(daily_df, tmp_path, config=None, ml_provider=None) -> PaperTradingEngine:
    config = config or Config()
    config.decision_thresholds.min_history_bars = 60

    def fetch(symbol, period, interval):
        return daily_df.copy()

    md = MarketDataProvider(fetch_fn=fetch)

    def fake_fundamentals_fetch(symbol):
        return {
            "revenue_growth": 0.1, "earnings_growth": 0.1, "eps": 5, "pe_ratio": 20, "pb_ratio": 3,
            "debt_to_equity": 50, "roe": 0.15, "profit_margin": 0.1, "operating_cash_flow": 100_000,
            "free_cash_flow": 50_000, "last_update": datetime.now(timezone.utc),
        }

    return PaperTradingEngine(
        config=config, market_data=md, fetch_news=False, fetch_social=False,
        fundamental_analyzer=FundamentalAnalyzer(fetch_fn=fake_fundamentals_fetch),
        journal=TradeJournal(path=str(tmp_path / "journal.csv")),
        decision_log=DecisionLog(path=str(tmp_path / "decisions.csv")),
        ml_provider=ml_provider,
    )


class _FixedMLProvider:
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def predict(self, symbol, daily):
        self.calls += 1
        return self.value


def _recent_start_date(n: int) -> str:
    return pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=n)[0].strftime("%Y-%m-%d")


def test_ml_probability_stays_none_without_a_provider(tmp_path):
    from tests.conftest import make_synthetic_ohlcv

    daily = make_synthetic_ohlcv(n=260, start_price=150.0, seed=51, start_date=_recent_start_date(260))
    engine = make_offline_engine(daily, tmp_path, ml_provider=None)  # today's default -- ML off
    result = engine.scan_symbol("NOMLCO")
    assert result is not None
    assert result.report.ml_probability_up is None
    assert engine.decision_log.all_entries()[0].ml_probability is None


def test_ml_probability_reaches_report_and_decision_log_when_provider_supplied(tmp_path):
    from tests.conftest import make_synthetic_ohlcv

    daily = make_synthetic_ohlcv(n=260, start_price=150.0, seed=53, start_date=_recent_start_date(260))
    fixed_provider = _FixedMLProvider(0.71)
    engine = make_offline_engine(daily, tmp_path, ml_provider=fixed_provider)
    result = engine.scan_symbol("MLCO")
    assert result is not None
    assert fixed_provider.calls == 1
    assert result.report.ml_probability_up == pytest.approx(0.71)
    assert engine.decision_log.all_entries()[0].ml_probability == pytest.approx(0.71)


def test_ml_provider_failure_treated_as_unavailable_not_a_crash(tmp_path):
    from tests.conftest import make_synthetic_ohlcv

    class _BrokenProvider:
        def predict(self, symbol, daily):
            raise RuntimeError("simulated model failure")

    daily = make_synthetic_ohlcv(n=260, start_price=150.0, seed=59, start_date=_recent_start_date(260))
    engine = make_offline_engine(daily, tmp_path, ml_provider=_BrokenProvider())
    result = engine.scan_symbol("BROKENMLCO")  # must not raise
    assert result is not None
    assert result.report.ml_probability_up is None
