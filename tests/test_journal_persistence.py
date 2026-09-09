"""
Phase 20 regression tests: account state (broker cash/positions,
CapitalProtection, the re-entry guard) must survive a process restart by
being reconstructed from the journal's persisted CSV history, since every
real `main.py paper-trade` run is a single one-shot process, not a
long-lived one.
"""

from __future__ import annotations

import csv
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from config.settings import Config
from data.market_data import MarketDataProvider
from paper_trading.engine import PaperTradingEngine
from paper_trading.journal import JournalEntry, TradeJournal


def make_fixed_quote_md(price: float) -> MarketDataProvider:
    def fetch(symbol, period, interval):
        return pd.DataFrame({
            "Open": [price], "High": [price], "Low": [price], "Close": [price], "Volume": [1_000_000],
        }, index=pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=1))
    return MarketDataProvider(fetch_fn=fetch)


# --- TradeJournal round-trip -------------------------------------------

def test_journal_reloads_open_and_closed_entries_from_existing_csv(tmp_path):
    path = str(tmp_path / "journal.csv")
    j1 = TradeJournal(path=path)
    open_entry = JournalEntry(
        trade_id="o1", symbol="OPENCO", side="BUY", entry_time=datetime.now(timezone.utc),
        entry_price=50.0, stop_loss=45.0, target=60.0, quantity=20, fees=1.0, sector="PHARMA",
    )
    j1.record_open(open_entry)
    closed_entry = JournalEntry(
        trade_id="c1", symbol="CLOSEDCO", side="BUY", entry_time=datetime.now(timezone.utc),
        entry_price=100.0, stop_loss=95.0, target=110.0, quantity=10, fees=1.0, sector="IT",
    )
    j1.record_open(closed_entry)
    j1.record_close("c1", datetime.now(timezone.utc), exit_price=108.0, exit_reason="TARGET",
                     fees=1.08, gross_pnl=80.0, net_pnl=77.92)

    # Simulate a fresh process: a NEW TradeJournal instance at the same path.
    j2 = TradeJournal(path=path)
    assert len(j2.open_trades()) == 1
    assert len(j2.closed_trades()) == 1
    reloaded_open = j2.open_trades()[0]
    assert reloaded_open.symbol == "OPENCO"
    assert reloaded_open.quantity == 20
    assert reloaded_open.sector == "PHARMA"
    assert isinstance(reloaded_open.entry_time, datetime)
    reloaded_closed = j2.closed_trades()[0]
    assert reloaded_closed.net_pnl == pytest.approx(77.92)
    assert reloaded_closed.sector == "IT"
    assert isinstance(reloaded_closed.exit_time, datetime)


def test_journal_loads_old_csv_without_sector_column(tmp_path):
    """Backward compatibility: the real logs/paper_trade_journal.csv
    accumulated before this fix has no `sector` column at all."""
    path = str(tmp_path / "old_journal.csv")
    old_fieldnames = [
        "trade_id", "symbol", "side", "entry_time", "entry_price", "stop_loss", "target",
        "quantity", "exit_time", "exit_price", "exit_reason", "fees", "slippage_estimate",
        "gross_pnl", "net_pnl", "confidence_at_entry", "decision_reasons",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=old_fieldnames)
        writer.writeheader()
        writer.writerow({
            "trade_id": "old1", "symbol": "OLDCO", "side": "BUY",
            "entry_time": datetime.now(timezone.utc).isoformat(), "entry_price": "100.0",
            "stop_loss": "95.0", "target": "110.0", "quantity": "5",
            "exit_time": "", "exit_price": "", "exit_reason": "", "fees": "0.5",
            "slippage_estimate": "0.0", "gross_pnl": "", "net_pnl": "",
            "confidence_at_entry": "", "decision_reasons": "",
        })

    journal = TradeJournal(path=path)
    assert len(journal.open_trades()) == 1
    entry = journal.open_trades()[0]
    assert entry.symbol == "OLDCO"
    assert entry.sector is None

    # The file must be migrated to the CURRENT header immediately, so the
    # next _append_row() (new-schema column order/count) doesn't misalign
    # against a stale old-schema header row.
    from paper_trading.journal import FIELDNAMES
    with open(path, newline="") as f:
        migrated_header = next(csv.reader(f))
    assert migrated_header == FIELDNAMES

    # A fresh append after migration must round-trip correctly.
    journal.record_open(JournalEntry(
        trade_id="new1", symbol="NEWCO", side="SELL", entry_time=datetime.now(timezone.utc),
        entry_price=200.0, stop_loss=210.0, target=180.0, quantity=3, fees=0.2, sector="AUTO",
    ))
    journal2 = TradeJournal(path=path)
    by_id = {e.trade_id: e for e in journal2.open_trades()}
    assert by_id["old1"].sector is None
    assert by_id["new1"].sector == "AUTO"


# --- PaperTradingEngine restoration -------------------------------------

def test_engine_restores_cash_positions_and_capital_state_after_restart(tmp_path):
    journal_path = str(tmp_path / "journal.csv")
    config = Config()
    md = make_fixed_quote_md(100.0)

    engine1 = PaperTradingEngine(config=config, market_data=md, journal=TradeJournal(path=journal_path))
    starting_cash = engine1.broker.get_balance()

    yesterday = datetime.now(timezone.utc) - timedelta(days=1)
    closed_entry = JournalEntry(
        trade_id="closed-1", symbol="CLOSEDCO", side="BUY", entry_time=yesterday,
        entry_price=100.0, stop_loss=95.0, target=110.0, quantity=10, fees=1.0, sector="IT",
    )
    engine1.journal.record_open(closed_entry)
    engine1.journal.record_close("closed-1", yesterday + timedelta(hours=2), exit_price=108.0,
                                  exit_reason="TARGET", fees=1.08, gross_pnl=80.0, net_pnl=77.92)

    open_entry = JournalEntry(
        trade_id="open-1", symbol="OPENCO", side="BUY",
        entry_time=datetime.now(timezone.utc) - timedelta(hours=20),
        entry_price=50.0, stop_loss=45.0, target=60.0, quantity=20, fees=1.0, sector="PHARMA",
    )
    engine1.journal.record_open(open_entry)

    # Simulate a restart: a brand-new engine + a brand-new TradeJournal
    # pointed at the same CSV path.
    engine2 = PaperTradingEngine(config=config, market_data=md, journal=TradeJournal(path=journal_path))

    expected_cash = starting_cash + 77.92 - (50.0 * 20 + 1.0)
    assert engine2.broker.get_balance() == pytest.approx(expected_cash)

    positions = {p.symbol: p for p in engine2.broker.get_positions()}
    assert "CLOSEDCO" not in positions
    assert "OPENCO" in positions
    assert positions["OPENCO"].quantity == 20
    assert positions["OPENCO"].side == "LONG"
    assert positions["OPENCO"].sector == "PHARMA"
    assert positions["OPENCO"].stop_loss == pytest.approx(45.0)

    assert engine2.capital_protection.state.open_positions == 1
    assert engine2.capital_protection.state.sector_exposure.get("PHARMA", 0.0) == pytest.approx(1000.0)
    assert engine2.capital_protection.state.current_capital == pytest.approx(
        config.paper_trading.starting_capital + 77.92
    )

    # The Phase-17 re-entry guard must survive the restart -- the engine
    # must "remember" OPENCO already has an open paper position.
    assert engine2._trade_ids.get("OPENCO") == "open-1"


def test_engine_can_manage_a_restored_position(tmp_path):
    """A position reconstructed from a prior run's journal must be a real,
    live Position that manage_open_positions() can auto-close on a
    stop/target hit -- not just a cosmetic reconstruction."""
    journal_path = str(tmp_path / "journal.csv")
    config = Config()
    md = make_fixed_quote_md(100.0)  # current quote already above target

    engine1 = PaperTradingEngine(config=config, market_data=md, journal=TradeJournal(path=journal_path))
    open_entry = JournalEntry(
        trade_id="open-1", symbol="OPENCO", side="BUY",
        entry_time=datetime.now(timezone.utc) - timedelta(hours=20),
        entry_price=50.0, stop_loss=45.0, target=60.0, quantity=20, fees=1.0, sector="PHARMA",
    )
    engine1.journal.record_open(open_entry)

    engine2 = PaperTradingEngine(config=config, market_data=md, journal=TradeJournal(path=journal_path))
    engine2.manage_open_positions()   # quote (100) is above target (60) -- should auto-close

    assert engine2.broker.get_positions() == []
    assert "OPENCO" not in engine2._trade_ids
    closed = [e for e in engine2.journal.closed_trades() if e.trade_id == "open-1"]
    assert len(closed) == 1
    assert closed[0].net_pnl > 0
    assert engine2.capital_protection.state.sector_exposure.get("PHARMA", 0.0) == pytest.approx(0.0)


def test_engine_restore_weekly_pnl_carries_over_across_a_restart(tmp_path):
    """A trade closed earlier in the current week must still count toward
    weekly P&L after a restart -- proving the chronological register_close
    replay, not a naive sum, is what reconstructs CapitalProtection."""
    journal_path = str(tmp_path / "journal.csv")
    config = Config()
    md = make_fixed_quote_md(100.0)

    today = date.today()
    week_start = today - timedelta(days=today.weekday())
    prior_day = week_start  # Monday of the current week -- always in-week

    engine1 = PaperTradingEngine(config=config, market_data=md, journal=TradeJournal(path=journal_path))
    exit_dt = datetime.combine(prior_day, datetime.min.time(), tzinfo=timezone.utc) + timedelta(hours=6)
    entry = JournalEntry(
        trade_id="c1", symbol="WKCO", side="BUY", entry_time=exit_dt - timedelta(hours=1),
        entry_price=100.0, stop_loss=95.0, target=110.0, quantity=10, fees=0.5, sector="IT",
    )
    engine1.journal.record_open(entry)
    engine1.journal.record_close("c1", exit_dt, exit_price=110.0, exit_reason="TARGET",
                                  fees=0.55, gross_pnl=100.0, net_pnl=98.95)

    engine2 = PaperTradingEngine(config=config, market_data=md, journal=TradeJournal(path=journal_path))
    assert engine2.capital_protection.state.weekly_pnl == pytest.approx(98.95)

    # A real cycle's first pre_trade_check (called for every scanned symbol,
    # unconditionally, via account_check in scan_symbol) rolls state.day
    # forward to real "today" and -- if that trade didn't close today --
    # resets daily_pnl to 0. Simulated directly here rather than driving a
    # full scan, since pre_trade_check is the exact real call site.
    engine2.capital_protection.pre_trade_check(symbol="ANY", sector=None, notional_exposure=0.0, today=today)
    assert engine2.capital_protection.state.day == today
    if prior_day != today:
        assert engine2.capital_protection.state.daily_pnl == pytest.approx(0.0)
    # Weekly P&L must survive the day-roll (still the same week).
    assert engine2.capital_protection.state.weekly_pnl == pytest.approx(98.95)


def test_fresh_engine_with_empty_journal_is_unaffected(tmp_path):
    """Regression guard: restoration must be a true no-op for a brand-new
    account (every existing test in this suite relies on this)."""
    journal_path = str(tmp_path / "journal.csv")
    config = Config()
    md = make_fixed_quote_md(100.0)
    engine = PaperTradingEngine(config=config, market_data=md, journal=TradeJournal(path=journal_path))
    assert engine.broker.get_balance() == pytest.approx(config.paper_trading.starting_capital)
    assert engine.broker.get_positions() == []
    assert engine.capital_protection.state.open_positions == 0
    assert engine._trade_ids == {}
