"""
Historical data providers for backtesting -- fundamentals, news, social
sentiment, and ML probability, all as of a specific point in time.

Why this file exists: free, point-in-time-correct historical datasets for
fundamentals, news, and social sentiment are not available in this
environment (and are hard to get anywhere without a paid vendor and careful
survivorship/restatement handling). Rather than fabricate plausible-looking
history for these components, the DEFAULT behavior for all three is to
report "unavailable" (return None) for every date. `strategy.signal_engine`
already treats `None` as "exclude this component and redistribute its
weight" rather than "assume neutral" -- see that module's docstring.

If you DO have a real, point-in-time-correct dataset (e.g. a fundamentals
vendor feed with as-of timestamps, or an archived news corpus), implement
`HistoricalFundamentalsProvider` / `HistoricalNewsProvider` /
`HistoricalSocialProvider` against it and pass your instance into
`Backtester`. The contract each provider MUST honor is in its docstring
below: never return information that would not have been known as of
`as_of`.

ML is different: it's not "unavailable by default", it's "only usable when
demonstrably trained out-of-sample." `HistoricalMLProvider` wraps an
already-fitted `MLBaselineModel` (fit on data strictly before `cutoff_date`,
e.g. a walk-forward fold's training window) and a leakage-safe feature
frame, and refuses -- hard -- to return a prediction for any date at or
before `cutoff_date`, regardless of what the caller asks for.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

import pandas as pd

from fundamentals.fundamental_analysis import FundamentalSnapshot
from models.ml_baseline import (
    TRADE_OUTCOME_FEATURE_COLS, MLBaselineModel, build_feature_matrix, build_trade_outcome_features,
)
from news.news_analysis import NewsAggregate
from sentiment.social_sentiment import SocialAggregate


class HistoricalFundamentalsProvider(ABC):
    @abstractmethod
    def get(self, symbol: str, as_of: pd.Timestamp) -> Optional[FundamentalSnapshot]:
        """Must return only information that would have been publicly known
        as of `as_of` (e.g. the most recently FILED quarterly report as of
        that date, not a report that was filed later but covers an earlier
        period). Return None if no point-in-time data is available."""
        raise NotImplementedError


class NoHistoricalFundamentalsProvider(HistoricalFundamentalsProvider):
    """Default: explicitly unavailable for every date. This is the honest
    default in the absence of a real point-in-time fundamentals dataset --
    NOT a stand-in that fabricates neutral or positive fundamentals."""

    def get(self, symbol: str, as_of: pd.Timestamp) -> Optional[FundamentalSnapshot]:
        return None


class HistoricalNewsProvider(ABC):
    @abstractmethod
    def get(self, symbol: str, as_of: pd.Timestamp) -> Optional[NewsAggregate]:
        """Must return only news items published on or before `as_of`."""
        raise NotImplementedError


class NoHistoricalNewsProvider(HistoricalNewsProvider):
    """Default: explicitly unavailable for every date -- never fabricated
    neutral/positive sentiment."""

    def get(self, symbol: str, as_of: pd.Timestamp) -> Optional[NewsAggregate]:
        return None


class HistoricalSocialProvider(ABC):
    @abstractmethod
    def get(self, symbol: str, as_of: pd.Timestamp) -> Optional[SocialAggregate]:
        """Must return only posts/mentions dated on or before `as_of`."""
        raise NotImplementedError


class NoHistoricalSocialProvider(HistoricalSocialProvider):
    """Default: explicitly unavailable for every date."""

    def get(self, symbol: str, as_of: pd.Timestamp) -> Optional[SocialAggregate]:
        return None


class HistoricalMLProvider:
    """
    Wraps an MLBaselineModel that has already been fit on data strictly
    before `cutoff_date` (e.g. train+validation bars of one walk-forward
    fold) and serves point-in-time predictions for dates AFTER that cutoff
    only. Any request at or before the cutoff returns None, regardless of
    whether features exist for that date -- this is a hard guard against
    accidentally scoring the model's own training window as if it were
    out-of-sample.

    `features` should be the leakage-safe frame from
    `models.ml_baseline.build_feature_matrix` computed over the FULL series
    (train+test) once, since each row of that frame already only uses data
    up to and including its own date -- computing it once is not itself a
    leakage risk, only *using* rows at/before the cutoff would be, which
    this class refuses to do.
    """

    def __init__(self, model: MLBaselineModel, cutoff_date: pd.Timestamp, features: pd.DataFrame):
        self.model = model
        self.cutoff_date = pd.Timestamp(cutoff_date)
        self.features = features

    def get(self, symbol: str, as_of: pd.Timestamp) -> Optional[float]:
        as_of = pd.Timestamp(as_of)
        if as_of <= self.cutoff_date:
            return None
        if as_of not in self.features.index:
            return None
        return self.model.predict_proba_up(self.features.loc[as_of])

    @classmethod
    def fit_on_training_window(
        cls, train_daily: pd.DataFrame, full_daily: pd.DataFrame, horizon: int = 5, n_splits: int = 5,
        min_auc: float = 0.53, label_mode: str = "direction",
    ) -> Optional["HistoricalMLProvider"]:
        """
        Convenience constructor: fits an MLBaselineModel on `train_daily`
        only, builds leakage-safe features over `full_daily` (so later dates
        can be scored), and returns a provider whose cutoff is
        `train_daily`'s last date -- or None if the model isn't usable
        (AUC too close to a coin flip; see MLBaselineModel.is_usable), so
        callers don't wire in a model that isn't adding real information.

        `label_mode`: `"direction"` (default) predicts next-`horizon`-day
        direction via `build_feature_matrix` -- this is the ORIGINAL,
        unchanged behavior every existing caller/test relies on.
        `"trade_outcome"` (spec Part 7) instead fits a triple-barrier
        target-before-stop classifier via `build_trade_outcome_features`
        (`horizon` is ignored in this mode -- the label's own
        `max_holding_days` governs how far forward it looks).
        """
        if label_mode == "trade_outcome":
            train_features = build_trade_outcome_features(train_daily)
            if len(train_features) < max(50, n_splits * 10):
                return None
            model = MLBaselineModel(n_splits=n_splits, feature_cols=TRADE_OUTCOME_FEATURE_COLS)
            model.fit(train_features)
            if not model.is_usable(min_auc=min_auc):
                return None
            full_features = build_trade_outcome_features(full_daily)
            cutoff = train_daily.index[-1]
            return cls(model=model, cutoff_date=cutoff, features=full_features)

        train_features = build_feature_matrix(train_daily, horizon=horizon)
        if len(train_features) < max(50, n_splits * 10):
            return None  # not enough data to fit or validate meaningfully
        model = MLBaselineModel(n_splits=n_splits)
        model.fit(train_features)
        if not model.is_usable(min_auc=min_auc):
            return None
        full_features = build_feature_matrix(full_daily, horizon=horizon)
        cutoff = train_daily.index[-1]
        return cls(model=model, cutoff_date=cutoff, features=full_features)
