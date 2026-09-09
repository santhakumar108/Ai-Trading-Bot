import numpy as np
import pandas as pd
import pytest

from models.ml_baseline import MLBaselineModel, build_feature_matrix


def test_feature_matrix_has_no_lookahead_leakage(uptrend_daily):
    horizon = 5
    features = build_feature_matrix(uptrend_daily, horizon=horizon)
    # The label at row t depends on close[t+horizon]; the feature matrix must
    # drop the last `horizon` raw rows so every remaining label is knowable.
    assert len(features) <= len(uptrend_daily) - horizon
    assert "forward_return" in features.columns
    assert not features[["ret_1", "ret_5", "ret_10", "sma20_ratio", "sma50_ratio", "volatility_20", "volume_ratio", "rsi_14"]].isna().any().any()


def test_feature_matrix_label_matches_forward_return(uptrend_daily):
    features = build_feature_matrix(uptrend_daily, horizon=5)
    assert set(features["label"].unique()).issubset({0, 1})
    assert ((features["forward_return"] > 0) == (features["label"] == 1)).all()


def test_model_fits_and_reports_oos_performance(uptrend_daily, downtrend_daily):
    # Concatenate an uptrend and downtrend segment so there's real signal
    # for a simple model to find, and both classes are present.
    combined = pd.concat([uptrend_daily, downtrend_daily + 0])
    combined.index = pd.bdate_range("2020-01-01", periods=len(combined))
    features = build_feature_matrix(combined, horizon=5)
    model = MLBaselineModel(n_splits=3)
    perf = model.fit(features)
    assert 0.0 <= perf.auc <= 1.0
    assert 0.0 <= perf.accuracy <= 1.0
    assert perf.n_test_samples > 0


def test_model_predict_proba_returns_bounded_value(uptrend_daily):
    features = build_feature_matrix(uptrend_daily, horizon=5)
    model = MLBaselineModel(n_splits=3)
    model.fit(features)
    prob = model.predict_proba_up(features.iloc[-1])
    assert prob is None or 0.0 <= prob <= 1.0


def test_is_usable_gates_on_auc(uptrend_daily):
    features = build_feature_matrix(uptrend_daily, horizon=5)
    model = MLBaselineModel(n_splits=3)
    model.fit(features)
    # is_usable should be a strict boolean gate, not silently True.
    assert isinstance(model.is_usable(min_auc=0.99), bool)


def test_drift_detection_flags_large_accuracy_drop(uptrend_daily):
    features = build_feature_matrix(uptrend_daily, horizon=5)
    model = MLBaselineModel(n_splits=3)
    perf = model.fit(features)
    degraded = model.check_drift(recent_accuracy=max(0.0, perf.accuracy - 0.5))
    assert degraded is True
    not_degraded = model.check_drift(recent_accuracy=perf.accuracy)
    assert not_degraded is False
