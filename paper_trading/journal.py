"""
Trade Journal.

A complete, append-only record of every paper trade: entry, exit, stop,
target, fees, slippage, and realized P&L. Persisted as CSV (human-readable,
easy to audit or import into a spreadsheet) under logs/, and also queryable
in-memory for the dashboard.
"""

from __future__ import annotations

import csv
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import List, Optional

DEFAULT_JOURNAL_PATH = os.path.join("logs", "paper_trade_journal.csv")


@dataclass
class JournalEntry:
    trade_id: str
    symbol: str
    side: str
    entry_time: datetime
    entry_price: float
    stop_loss: float
    target: float
    quantity: int
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None   # "TARGET", "STOP", "MANUAL", "TIMEOUT"
    fees: float = 0.0
    slippage_estimate: float = 0.0
    gross_pnl: Optional[float] = None
    net_pnl: Optional[float] = None
    confidence_at_entry: Optional[float] = None
    decision_reasons: str = ""
    sector: Optional[str] = None


FIELDNAMES = list(JournalEntry.__dataclass_fields__.keys())

_OPTIONAL_STR_FIELDS = {"exit_reason", "sector"}
_FLOAT_FIELDS = {"entry_price", "stop_loss", "target", "exit_price", "fees",
                  "slippage_estimate", "gross_pnl", "net_pnl", "confidence_at_entry"}
_REQUIRED_FLOAT_FIELDS = {"entry_price", "stop_loss", "target", "fees", "slippage_estimate"}
_DATETIME_FIELDS = {"entry_time", "exit_time"}


def _blank(value: Optional[str]) -> bool:
    return value is None or value == "" or value == "None"


def _parse_row(row: dict) -> JournalEntry:
    """Reconstructs a JournalEntry from one CSV row (a `csv.DictReader`
    dict keyed by the FILE'S OWN header -- not a hardcoded FIELDNAMES zip
    -- so an older journal file written before `sector` existed still
    loads correctly, with sector=None for those rows)."""
    kwargs: dict = {}
    for key in JournalEntry.__dataclass_fields__:
        raw = row.get(key)
        if key == "quantity":
            kwargs[key] = int(float(raw))
        elif key in _DATETIME_FIELDS:
            kwargs[key] = None if _blank(raw) else datetime.fromisoformat(raw)
        elif key in _FLOAT_FIELDS:
            if _blank(raw):
                kwargs[key] = 0.0 if key in _REQUIRED_FLOAT_FIELDS else None
            else:
                kwargs[key] = float(raw)
        elif key in _OPTIONAL_STR_FIELDS:
            kwargs[key] = None if _blank(raw) else str(raw)
        elif key == "decision_reasons":
            kwargs[key] = "" if raw is None else str(raw)
        else:
            kwargs[key] = raw
    return JournalEntry(**kwargs)


class TradeJournal:
    def __init__(self, path: str = DEFAULT_JOURNAL_PATH):
        self.path = path
        self._entries: List[JournalEntry] = []
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(FIELDNAMES)
        else:
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                file_fieldnames = reader.fieldnames or []
                for row in reader:
                    self._entries.append(_parse_row(row))
            # Migrate an older file (e.g. one written before `sector` was
            # added to JournalEntry) to the current schema immediately --
            # otherwise a future `_append_row()` (writing FIELDNAMES-order
            # columns) would misalign against this file's stale header,
            # silently corrupting the NEXT append, not just old rows.
            if file_fieldnames != FIELDNAMES:
                self._rewrite()

    def record_open(self, entry: JournalEntry) -> None:
        self._entries.append(entry)
        self._append_row(entry)

    def record_close(
        self, trade_id: str, exit_time: datetime, exit_price: float, exit_reason: str,
        fees: float, gross_pnl: float, net_pnl: float,
    ) -> None:
        for e in self._entries:
            if e.trade_id == trade_id and e.exit_time is None:
                e.exit_time = exit_time
                e.exit_price = exit_price
                e.exit_reason = exit_reason
                e.fees += fees
                e.gross_pnl = gross_pnl
                e.net_pnl = net_pnl
                break
        self._rewrite()

    def open_trades(self) -> List[JournalEntry]:
        return [e for e in self._entries if e.exit_time is None]

    def closed_trades(self) -> List[JournalEntry]:
        return [e for e in self._entries if e.exit_time is not None]

    def net_pnls(self) -> List[float]:
        return [e.net_pnl for e in self.closed_trades() if e.net_pnl is not None]

    def win_rate(self) -> float:
        pnls = self.net_pnls()
        if not pnls:
            return 0.0
        wins = [p for p in pnls if p > 0]
        return len(wins) / len(pnls)

    def profit_factor(self) -> float:
        pnls = self.net_pnls()
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = -sum(p for p in pnls if p < 0)
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    def _append_row(self, entry: JournalEntry) -> None:
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([getattr(entry, k) for k in FIELDNAMES])

    def _rewrite(self) -> None:
        with open(self.path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(FIELDNAMES)
            for e in self._entries:
                writer.writerow([getattr(e, k) for k in FIELDNAMES])
