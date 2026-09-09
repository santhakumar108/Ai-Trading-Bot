"""
Phase 20: the customer-facing HTML generator must show REAL numbers when
real data exists, and fall back to a clearly-labeled SAMPLE section (never
fabricated as real) when it doesn't -- same contract already validated
for the interactive artifact this mirrors.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from config.settings import Config
from data.market_data import MarketDataProvider
from dashboard.customer_report import render_customer_html
from paper_trading.engine import PaperTradingEngine
from paper_trading.journal import JournalEntry, TradeJournal


def make_fixed_quote_md(price: float) -> MarketDataProvider:
    def fetch(symbol, period, interval):
        return pd.DataFrame({
            "Open": [price], "High": [price], "Low": [price], "Close": [price], "Volume": [1_000_000],
        }, index=pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=1))
    return MarketDataProvider(fetch_fn=fetch)


def test_period_pnl_summary_buckets_by_day_week_month(tmp_path):
    from dashboard.customer_report import _period_pnl_summary

    now_ist = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
    today = now_ist.date()
    week_start = today - timedelta(days=today.weekday())
    prior_month_day = today.replace(day=1) - timedelta(days=1)  # last day of the PREVIOUS month

    def _closed(trade_id, exit_date, pnl):
        exit_dt = datetime.combine(exit_date, datetime.min.time(), tzinfo=timezone.utc) + timedelta(hours=10)
        return JournalEntry(
            trade_id=trade_id, symbol="X", side="BUY", entry_time=exit_dt - timedelta(hours=1),
            entry_price=100.0, stop_loss=95.0, target=110.0, quantity=1,
            exit_time=exit_dt, exit_price=110.0, exit_reason="TARGET",
            fees=0.0, gross_pnl=pnl, net_pnl=pnl,
        )

    closed = [
        _closed("t1", today, 50.0),
        _closed("t2", week_start, 30.0),        # in-week (Monday); counted in week+month unless today IS Monday, where it's also "today"
        _closed("t3", prior_month_day, 999.0),  # previous month -- must be excluded from today/week/month
    ]
    summary = _period_pnl_summary(closed, now_ist)

    today_pnl, today_n = summary["Today"]
    expected_today_pnl = 50.0 + (30.0 if week_start == today else 0.0)
    expected_today_n = 1 + (1 if week_start == today else 0)
    assert today_pnl == pytest.approx(expected_today_pnl)
    assert today_n == expected_today_n

    # t1 and t2 are always in the current week; t3 (previous month) never is.
    week_pnl, week_n = summary["This Week"]
    assert week_pnl == pytest.approx(80.0)
    assert week_n == 2

    # t3 (previous month) is always excluded from "this month"; t2
    # (week_start) is only in the current month if the current week
    # doesn't cross a month boundary.
    month_start = today.replace(day=1)
    t2_in_month = week_start >= month_start
    expected_month_pnl = 50.0 + (30.0 if t2_in_month else 0.0)
    expected_month_n = 1 + (1 if t2_in_month else 0)
    month_pnl, month_n = summary["This Month"]
    assert month_pnl == pytest.approx(expected_month_pnl)
    assert month_n == expected_month_n

    all_pnl, all_n = summary["All Time"]
    assert all_pnl == pytest.approx(1079.0)
    assert all_n == 3


def test_empty_account_shows_empty_state_and_sample_labels(tmp_path):
    config = Config()
    md = make_fixed_quote_md(100.0)
    engine = PaperTradingEngine(config=config, market_data=md, journal=TradeJournal(path=str(tmp_path / "journal.csv")))
    html = render_customer_html(engine, config)
    assert "0 holdings today" in html
    assert "0 trades today" in html
    assert html.count("Sample") >= 2  # holdings + activity fallback banners
    assert "&#8377;10,00,000" in html  # starting capital, Indian-grouped


def test_real_holding_and_trade_render_as_real_not_sample(tmp_path):
    config = Config()
    md = make_fixed_quote_md(120.0)  # current quote above the entry, for a real gain
    engine = PaperTradingEngine(config=config, market_data=md, journal=TradeJournal(path=str(tmp_path / "journal.csv")))

    open_entry = JournalEntry(
        trade_id="o1", symbol="RELIANCE.NS", side="BUY",
        entry_time=datetime.now(timezone.utc) - timedelta(hours=5),
        entry_price=100.0, stop_loss=90.0, target=150.0, quantity=10, fees=1.0, sector="ENERGY",
    )
    engine.journal.record_open(open_entry)
    # Directly seed the broker with the matching open position -- the same
    # end state PaperTradingEngine._restore_state_from_journal would leave
    # after a restart, without re-running __init__ here.
    from broker.broker_interface import Position
    engine.broker._positions["RELIANCE.NS"] = Position(
        symbol="RELIANCE.NS", side="LONG", quantity=10, average_price=100.0,
        stop_loss=90.0, target=150.0, sector="ENERGY",
    )

    html = render_customer_html(engine, config)
    assert "RELIANCE" in html
    # The real holding must render inside the real-data block...
    real_section_start = html.index('<div class="real-block">')
    reliance_index = html.index("RELIANCE", real_section_start)
    assert reliance_index > real_section_start
    # ...and the holdings SAMPLE fallback must be absent now that real data exists.
    assert "The exact format each holding will use" not in html
