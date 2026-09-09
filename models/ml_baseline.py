"""
Machine Learning Baseline (spec section 15).

Principles applied here:
  * Start with an interpretable baseline (logistic regression) before ever
    reaching for anything fancier -- there is no deep-learning model in
    this system, and none should be added "because it sounds advanced"
    without first showing it beats this baseline out-of-sample.
  * No data leakage: every feature at row t is computed using only data
    available up to and including bar t; the label is the FORWARD return
    over the next `horizon` bars, and the last `horizon` rows are dropped
    (their label would require future data that doesn't exist yet).
  * Time-series cross-validation only (scikit-learn's TimeSeriesSplit) --
    never a random/shuffled split, which would leak future information into
    training folds.
  * Probability calibration via Platt scaling (sklearn's CalibratedClassifierCV
    with a held-out slice), so the model's output can be read as an
    approximate probability rather than an arbitrary score.
  * Out-of-sample performance (AUC, accuracy, Brier score) is always
    reported alongside the trained model; a model that isn't clearly better
    than a coin flip out-of-sample should not be wired into the signal
    engine's `ml_probability_up` input.
  * Model drift monitoring: `MLBaselineModel.check_drift` compares recent
    realized accuracy against the training-time out-of-sample accuracy and
    flags when it has degraded materially, which is the trigger for a
    defined retrain procedure -- not silent, automatic retraining.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

DIRECTION_FEATURE_COLS = [
    "ret_1", "ret_5", "ret_10", "sma20_ratio", "sma50_ratio", "volatility_20", "volume_ratio", "rsi_14",
]

# Spec Part 7: "features can include alpha scores, regime, technical
# indicators" -- adds ADX (trend strength) and MACD-histogram momentum on
# top of the direction model's base feature set, plus a numeric encoding of
# the market regime at that bar (data/market_data.py's classify_regime,
# same trailing-only/point-in-time-safe classifier Phase 2 wired into live
# scanning). Exported so callers (HistoricalMLProvider, LiveTradeOutcomeMLProvider)
# can pass it as MLBaselineModel's feature_cols.
TRADE_OUTCOME_FEATURE_COLS = DIRECTION_FEATURE_COLS + [
    "adx_14", "macd_hist_norm", "regime_trend", "regime_vol",
]


def build_feature_matrix(daily: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    Builds a leakage-safe feature matrix + forward-looking label from daily
    OHLCV data. Every feature column uses only information available at
    time t; `label` is 1 if the close `horizon` bars later is higher than
    the close at t, else 0. The final `horizon` rows are dropped because
    their label is unknowable without future data.
    """
    df = daily.copy()
    close = df["Close"]

    df["ret_1"] = close.pct_change(1)
    df["ret_5"] = close.pct_change(5)
    df["ret_10"] = close.pct_change(10)
    df["sma20_ratio"] = close / close.rolling(20).mean() - 1
    df["sma50_ratio"] = close / close.rolling(50).mean() - 1
    df["volatility_20"] = np.log(close / close.shift(1)).rolling(20).std()
    df["volume_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = (100 - 100 / (1 + rs)).fillna(50.0)

    # Forward label -- the only place future data is touched, and it is
    # explicitly excluded from the feature set.
    df["forward_return"] = close.shift(-horizon) / close - 1
    df["label"] = (df["forward_return"] > 0).astype(int)

    df = df.dropna(subset=DIRECTION_FEATURE_COLS)
    if horizon > 0:
        df = df.iloc[:-horizon]  # drop rows whose label needs unseen future bars
    return df[DIRECTION_FEATURE_COLS + ["label", "forward_return"]]


def build_trade_outcome_features(
    daily: pd.DataFrame,
    atr_window: int = 14,
    stop_atr_multiple: float = 1.5,
    target_atr_multiple: float = 3.0,
    max_holding_days: int = 20,
) -> pd.DataFrame:
    """
    Spec Part 7: "the ML model should NOT simply predict whether tomorrow's
    price is green" -- builds a TRIPLE-BARRIER trade-OUTCOME label instead
    of `build_feature_matrix`'s next-day-direction one.

    For each bar t, projects an ATR-based stop/target from `close[t]`
    (assuming a LONG hypothesis -- this model answers "if I bought here,
    would the target or the stop be hit first", not "should I buy") and
    walks forward up to `max_holding_days` bars checking each day's
    High/Low against those levels. If BOTH the stop and target would be
    touched on the same forward bar, the STOP takes priority (the same
    conservative tie-break `backtesting/backtester.py`'s bar-by-bar exit
    logic already uses -- we can't know the true intraday order from daily
    OHLC, so this never labels a same-bar ambiguity as a win). label=1 if
    the target is hit first, 0 if the stop is hit first, and the row is
    DROPPED (not fabricated as either class) if NEITHER is hit within the
    horizon -- an honest "no resolved outcome" rather than a forced guess.

    Every FEATURE column (as opposed to the label) at row t uses only
    `daily.iloc[:t+1]` -- identical leakage-safety property to
    `build_feature_matrix`; only the label deliberately looks forward,
    exactly as a supervised-learning target must.
    """
    from indicators.technical import adx as _adx, macd as _macd, atr as _atr
    from data.market_data import classify_regime

    df = daily.copy()
    close = df["Close"]

    df["ret_1"] = close.pct_change(1)
    df["ret_5"] = close.pct_change(5)
    df["ret_10"] = close.pct_change(10)
    df["sma20_ratio"] = close / close.rolling(20).mean() - 1
    df["sma50_ratio"] = close / close.rolling(50).mean() - 1
    df["volatility_20"] = np.log(close / close.shift(1)).rolling(20).std()
    df["volume_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = (100 - 100 / (1 + rs)).fillna(50.0)

    df["adx_14"] = _adx(df["High"], df["Low"], close, atr_window)
    _, _, macd_hist = _macd(close)
    df["macd_hist_norm"] = (macd_hist / close).fillna(0.0)

    atr_s = _atr(df["High"], df["Low"], close, atr_window)

    n = len(df)
    regime_trend = np.zeros(n)
    regime_vol = np.zeros(n)
    outcome = np.full(n, np.nan)
    high = df["High"].values
    low = df["Low"].values
    close_v = close.values
    atr_v = atr_s.values

    for i in range(n):
        regime = classify_regime(df, i)  # trailing-only, point-in-time-safe (Phase 2)
        if regime.startswith("BULL"):
            regime_trend[i] = 1.0
        elif regime.startswith("BEAR"):
            regime_trend[i] = -1.0
        regime_vol[i] = 1.0 if regime.endswith("HIGH_VOL") else 0.0

        entry = close_v[i]
        a = atr_v[i]
        if np.isnan(a) or a <= 0 or i + 1 >= n:
            continue
        stop = entry - stop_atr_multiple * a
        target = entry + target_atr_multiple * a
        end = min(i + 1 + max_holding_days, n)
        for j in range(i + 1, end):
            hit_stop = low[j] <= stop
            hit_target = high[j] >= target
            if hit_stop:  # stop takes priority on a same-bar tie -- see docstring
                outcome[i] = 0.0
                break
            if hit_target:
                outcome[i] = 1.0
                break

    df["regime_trend"] = regime_trend
    df["regime_vol"] = regime_vol
    df["label"] = outcome

    df = df.dropna(subset=TRADE_OUTCOME_FEATURE_COLS + ["label"])
    return df[TRADE_OUTCOME_FEATURE_COLS + ["label"]]


@dataclass
class OutOfSamplePerformance:
    auc: float
    accuracy: float
    brier_score: float
    n_test_samples: int
    fold_scores: List[float] = field(default_factory=list)


class MLBaselineModel:
    """Thin wrapper around a calibrated logistic regression, with proper
    time-series validation baked in. Optional: the signal engine works
    without this model (ml_probability_up=None is a valid input meaning
    'no ML opinion') -- this exists to ADD one more independent, measured
    vote, not to replace the rest of the system."""

    def __init__(self, n_splits: int = 5, feature_cols: Optional[List[str]] = None):
        self.n_splits = n_splits
        self.model = None
        self.oos_performance: Optional[OutOfSamplePerformance] = None
        # Defaults to the original 8-column direction-model feature set --
        # every existing caller/test that doesn't pass feature_cols keeps
        # identical behavior. Pass TRADE_OUTCOME_FEATURE_COLS to fit on the
        # richer trade-outcome feature set instead (see
        # build_trade_outcome_features above).
        self.feature_cols = feature_cols if feature_cols is not None else list(DIRECTION_FEATURE_COLS)

    def fit(self, features: pd.DataFrame) -> OutOfSamplePerformance:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        X = features[self.feature_cols].values
        y = features["label"].values

        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        fold_aucs: List[float] = []
        all_probs = np.zeros(len(y)) * np.nan

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            class_counts = np.bincount(y_train.astype(int))
            if len(class_counts) < 2 or class_counts.min() < 2:
                # Too few examples of one class in this training fold to
                # calibrate (or fit) meaningfully -- skip rather than let a
                # heavily one-sided historical window crash the whole run.
                continue

            base_pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
            ])
            # Calibrate on a further internal split of the training fold only
            # (never touches X_test), preventing calibration leakage. cv is
            # capped by the rarer class's count so a lopsided fold (e.g. a
            # strong one-directional trend) doesn't crash calibration.
            calib_cv = int(min(3, class_counts.min()))
            calibrated = CalibratedClassifierCV(base_pipe, method="sigmoid", cv=calib_cv)
            calibrated.fit(X_train, y_train)

            probs = calibrated.predict_proba(X_test)[:, 1]
            all_probs[test_idx] = probs
            if len(np.unique(y_test)) > 1:
                fold_aucs.append(roc_auc_score(y_test, probs))

        valid_mask = ~np.isnan(all_probs)
        y_valid = y[valid_mask]
        probs_valid = all_probs[valid_mask]

        auc = float(roc_auc_score(y_valid, probs_valid)) if len(np.unique(y_valid)) > 1 else 0.5
        accuracy = float(accuracy_score(y_valid, (probs_valid > 0.5).astype(int))) if len(y_valid) else 0.0
        brier = float(brier_score_loss(y_valid, probs_valid)) if len(y_valid) else 1.0

        self.oos_performance = OutOfSamplePerformance(
            auc=auc, accuracy=accuracy, brier_score=brier,
            n_test_samples=int(valid_mask.sum()), fold_scores=fold_aucs,
        )

        # Final model fit on ALL data, for live/paper-trading inference.
        # Its own OOS performance was already measured above via CV, honestly,
        # before this refit -- this refit is not what the reported metrics
        # describe.
        final_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=1.0)),
        ])
        full_class_counts = np.bincount(y.astype(int))
        final_calib_cv = int(min(3, full_class_counts.min())) if len(full_class_counts) > 1 else 2
        final_calib_cv = max(final_calib_cv, 2)
        self.model = CalibratedClassifierCV(final_pipe, method="sigmoid", cv=final_calib_cv)
        self.model.fit(X, y)

        return self.oos_performance

    def is_usable(self, min_auc: float = 0.53) -> bool:
        """A baseline model with AUC barely above 0.5 is not adding real
        information -- gate its use rather than pretending it helps."""
        return self.oos_performance is not None and self.oos_performance.auc >= min_auc

    def predict_proba_up(self, features_row: pd.Series) -> Optional[float]:
        if self.model is None:
            return None
        X = features_row[self.feature_cols].values.reshape(1, -1)
        return float(self.model.predict_proba(X)[0, 1])

    def check_drift(self, recent_accuracy: float, degradation_threshold: float = 0.08) -> bool:
        """Returns True if recent realized accuracy has degraded materially
        vs. the training-time out-of-sample accuracy, signaling that a
        defined retrain procedure should be triggered (not automatic)."""
        if self.oos_performance is None:
            return False
        return (self.oos_performance.accuracy - recent_accuracy) > degradation_threshold
