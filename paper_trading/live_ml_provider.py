"""
Live ML trade-outcome provider (spec Part 7), opt-in via `--ml`.

Unlike backtesting's `HistoricalMLProvider` (backtesting/
historical_providers.py -- fit once per walk-forward fold on a FIXED past
training window), a live paper-trading scan has no fold structure: "now" is
always the most recent bar, so there's no lookahead risk in fitting on
whatever history is available at the moment of the scan. What live scanning
DOES need that backtesting doesn't is a CACHE -- fitting a calibrated
model on 1-5 years of history is not free, and `PaperTradingEngine.scan_symbol`
may be called for the same symbol many times across a session (each
`paper-trade` poll cycle, a dashboard refresh, etc.). Fitting fresh every
call would make every user's scan slower, including the many who never ask
for ML at all (which is why `PaperTradingEngine.ml_provider` defaults to
None -- see paper_trading/engine.py).

Never fabricates a probability: insufficient history, or a fitted model
whose out-of-sample AUC doesn't clear the configured gate, both resolve to
`None` -- exactly the "unavailable, excluded, weight redistributed" contract
`strategy/signal_engine.py` already enforces for every other optional
component.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import pandas as pd

from models.ml_baseline import TRADE_OUTCOME_FEATURE_COLS, MLBaselineModel, build_trade_outcome_features

logger = logging.getLogger(__name__)

# Sentinel cached in place of a model when fitting was attempted but the
# result wasn't usable (insufficient history, or AUC below the gate) --
# distinct from "never attempted", so an unusable symbol isn't re-fit on
# every single scan of it.
_UNUSABLE = object()


class LiveTradeOutcomeMLProvider:
    def __init__(
        self,
        min_history_bars: int = 300,
        min_auc: float = 0.53,
        stop_atr_multiple: float = 1.5,
        target_atr_multiple: float = 3.0,
        max_holding_days: int = 20,
        n_splits: int = 5,
    ):
        self.min_history_bars = min_history_bars
        self.min_auc = min_auc
        self.stop_atr_multiple = stop_atr_multiple
        self.target_atr_multiple = target_atr_multiple
        self.max_holding_days = max_holding_days
        self.n_splits = n_splits
        self._cache: Dict[str, object] = {}  # symbol -> MLBaselineModel | _UNUSABLE

    def predict(self, symbol: str, daily: pd.DataFrame) -> Optional[float]:
        """Returns a calibrated P(target-before-stop) for `symbol`'s most
        recent bar, or None if no usable model exists for it (fitting is
        attempted at most once per symbol per provider instance -- see
        `refresh` to force a re-fit)."""
        model = self._cache.get(symbol)
        if model is None:
            model = self._fit(symbol, daily)
            self._cache[symbol] = model if model is not None else _UNUSABLE
        if model is _UNUSABLE or model is None:
            return None
        features = build_trade_outcome_features(
            daily, stop_atr_multiple=self.stop_atr_multiple,
            target_atr_multiple=self.target_atr_multiple, max_holding_days=self.max_holding_days,
        )
        if features.empty:
            return None
        return model.predict_proba_up(features.iloc[-1])

    def refresh(self, symbol: str, daily: pd.DataFrame) -> Optional[float]:
        """Forces a re-fit for `symbol` (e.g. a scheduled retrain), bypassing
        the cache. Not wired to any automatic trigger -- an operator/caller
        decides when a retrain is warranted (spec Part 7's drift-monitoring
        intent; see MLBaselineModel.check_drift for the signal to watch)."""
        self._cache.pop(symbol, None)
        return self.predict(symbol, daily)

    def _fit(self, symbol: str, daily: pd.DataFrame) -> Optional[MLBaselineModel]:
        if len(daily) < self.min_history_bars:
            logger.info(
                "LiveTradeOutcomeMLProvider: %s has %d bars, need >= %d -- ML unavailable, not fabricated.",
                symbol, len(daily), self.min_history_bars,
            )
            return None
        features = build_trade_outcome_features(
            daily, stop_atr_multiple=self.stop_atr_multiple,
            target_atr_multiple=self.target_atr_multiple, max_holding_days=self.max_holding_days,
        )
        if len(features) < max(50, self.n_splits * 10):
            logger.info("LiveTradeOutcomeMLProvider: %s has too few resolved-outcome rows to fit.", symbol)
            return None
        model = MLBaselineModel(n_splits=self.n_splits, feature_cols=TRADE_OUTCOME_FEATURE_COLS)
        model.fit(features)
        if not model.is_usable(min_auc=self.min_auc):
            logger.info(
                "LiveTradeOutcomeMLProvider: %s model AUC below %.2f gate -- treated as unavailable.",
                symbol, self.min_auc,
            )
            return None
        return model
