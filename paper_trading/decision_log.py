"""
Decision Log (spec sections 17 & 20).

`TradeJournal` (paper_trading/journal.py) records trades that were actually
OPENED. That is not the full record spec section 17 asks for: "record
every signal, record rejected trades ... every NO TRADE decision should
include a reason." This module is the wider, append-only log of EVERY scan
decision `PaperTradingEngine.scan_symbol` makes -- BUY/SELL/HOLD/NO TRADE
alike, approved or rejected, including cases where there wasn't even enough
data to reach the signal engine (a data-quality or provider failure) -- so
nothing about why the system did or didn't act is ever silently discarded.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

DEFAULT_DECISION_LOG_PATH = os.path.join("logs", "decision_log.csv")


@dataclass
class DecisionLogEntry:
    timestamp: datetime
    symbol: str
    decision: str                  # "BUY" | "SELL" | "HOLD" | "NO TRADE"
    approved: bool
    confidence: float
    confidence_label: str
    direction: str
    data_quality_status: str       # "OK" | "DEGRADED" | "INVALID" | "UNKNOWN" (no data at all)
    reasons: str                   # every reason, joined with "; " -- never blank for a rejection
    sector: Optional[str] = None
    # -- Spec Part 18: full per-candidate schema, small-account upgrade --
    # Every field below is Optional and defaults to None when genuinely not
    # computed for this decision (e.g. no RiskAssessment was ever reached,
    # or -- for sector_score -- the underlying score simply doesn't exist
    # yet; see paper_trading/engine.py's _log_decision docstring). None
    # here always means "not available", never a fabricated 0.
    account_balance: Optional[float] = None
    price: Optional[float] = None
    affordable_quantity: Optional[int] = None
    technical_score: Optional[float] = None
    fundamental_score: Optional[float] = None
    market_score: Optional[float] = None
    sector_score: Optional[float] = None      # no sector-relative score exists anywhere in this
                                               # system (no sector-index data source) -- permanently None
    momentum_score: Optional[float] = None    # TechnicalSnapshot.momentum_score(); None only when no
                                               # TechnicalSnapshot was ever built for this decision (the
                                               # data-quality gate or TechnicalAnalyzer failed first --
                                               # see engine.py's _log_decision call sites)
    news_score: Optional[float] = None
    sentiment_score: Optional[float] = None
    volume_score: Optional[float] = None
    risk_score: Optional[float] = None
    ml_probability: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    expected_value: Optional[float] = None
    estimated_cost: Optional[float] = None
    position_size: Optional[int] = None
    rejection_reason: Optional[str] = None    # the FIRST/primary reason; `reasons` above has all of them
    market_regime: str = "UNKNOWN"            # spec Part 5.6/16 -- see data/market_data.py's classify_regime()
    # Spec Part 4/3, surfaced into the decision log (Phase 9 wiring):
    fundamental_risk: Optional[float] = None      # FundamentalSnapshot.fundamental_risk_score(), when available
    news_sentiment_class: Optional[str] = None    # NewsAggregate.sentiment_class, when available


FIELDNAMES = list(DecisionLogEntry.__dataclass_fields__.keys())


class DecisionLog:
    """Persisted as CSV under logs/ (same pattern as TradeJournal) --
    human-readable, easy to audit or load into a spreadsheet -- and also
    queryable in-memory for the dashboard/CLI within one process run."""

    def __init__(self, path: str = DEFAULT_DECISION_LOG_PATH):
        self.path = path
        self._entries: List[DecisionLogEntry] = []
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(FIELDNAMES)

    def record(self, entry: DecisionLogEntry) -> None:
        self._entries.append(entry)
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([getattr(entry, k) for k in FIELDNAMES])

    def all_entries(self) -> List[DecisionLogEntry]:
        return list(self._entries)

    def approved_entries(self) -> List[DecisionLogEntry]:
        return [e for e in self._entries if e.approved]

    def rejected_entries(self) -> List[DecisionLogEntry]:
        return [e for e in self._entries if not e.approved]

    def entries_for_symbol(self, symbol: str) -> List[DecisionLogEntry]:
        return [e for e in self._entries if e.symbol == symbol]

    def no_trade_count(self) -> int:
        return sum(1 for e in self._entries if e.decision == "NO TRADE")
