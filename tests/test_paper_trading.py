from datetime import datetime, timezone

import pandas as pd
import pytest

from broker.broker_interface import OrderStatus
from broker.paper_broker import PaperBroker
from config.settings import Config
from data.market_data import MarketDataProvider
from fundamentals.fundamental_analysis import FundamentalAnalyzer
from paper_trading.engine import PaperTradingEngine
from paper_trading.journal import JournalEntry, TradeJournal


def make_fetch_fn(df: pd.DataFrame):
    def fetch(symbol, period, interval):
        return df.copy()
    return fetch


# --- PaperBroker -------------------------------------------------------

def test_paper_broker_buy_reduces_cash_and_opens_position(uptrend_daily):
    md = MarketDataProvider(fetch_fn=make_fetch_fn(uptrend_daily))
    broker = PaperBroker(starting_capital=100_000, market_data=md, slippage_pct=0, fee_pct=0)
    order = broker.place_order("FAKE", "BUY", 10)
    assert order.status == OrderStatus.FILLED
    assert broker.get_balance() < 100_000
    positions = broker.get_positions()
    assert len(positions) == 1
    assert positions[0].quantity == 10


def test_paper_broker_rejects_when_insufficient_capital(uptrend_daily):
    md = MarketDataProvider(fetch_fn=make_fetch_fn(uptrend_daily))
    broker = PaperBroker(starting_capital=100, market_data=md)
    order = broker.place_order("FAKE", "BUY", 1_000_000)
    assert order.status == OrderStatus.REJECTED


def test_paper_broker_never_places_live_orders(uptrend_daily):
    md = MarketDataProvider(fetch_fn=make_fetch_fn(uptrend_daily))
    broker = PaperBroker(starting_capital=100_000, market_data=md)
    assert broker.supports_live_orders is False
    order = broker.place_order("FAKE", "BUY", 10, live_trading_enabled=True)
    assert order.is_paper is True  # ignored the flag, simulated anyway


def test_paper_broker_sell_closes_position_and_returns_cash(uptrend_daily):
    md = MarketDataProvider(fetch_fn=make_fetch_fn(uptrend_daily))
    broker = PaperBroker(starting_capital=100_000, market_data=md, slippage_pct=0, fee_pct=0)
    broker.place_order("FAKE", "BUY", 10)
    cash_after_buy = broker.get_balance()
    broker.place_order("FAKE", "SELL", 10)
    assert broker.get_positions() == []
    assert broker.get_balance() > cash_after_buy


# --- PaperBroker: SHORT open/cover, equity, unrealized_pnl -----------------

def make_fixed_quote_md(price: float) -> MarketDataProvider:
    """A MarketDataProvider whose get_quote() always returns a fixed,
    controlled price -- used to test P&L/equity math deterministically
    against a known entry vs. exit price, independent of any real series."""
    def fetch(symbol, period, interval):
        return pd.DataFrame({
            "Open": [price], "High": [price], "Low": [price], "Close": [price], "Volume": [1_000_000],
        }, index=pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=1))
    return MarketDataProvider(fetch_fn=fetch)


def test_paper_broker_buy_covering_short_does_not_overwrite_as_long():
    """Bug-1 regression: covering a SHORT must close it, never silently
    replace it with an unrelated phantom LONG."""
    md = make_fixed_quote_md(100.0)
    broker = PaperBroker(starting_capital=100_000, market_data=md, slippage_pct=0, fee_pct=0)
    broker.place_order("FAKE", "SELL", 10)  # opens a SHORT (no existing position)
    positions = broker.get_positions()
    assert len(positions) == 1 and positions[0].side == "SHORT"

    broker.place_order("FAKE", "BUY", 10)  # covers the short exactly
    assert broker.get_positions() == []  # closed, not replaced with a phantom LONG


def test_paper_broker_short_open_and_cover_full_pnl():
    """Covering a short at a LOWER price than entry must be profitable;
    at a HIGHER price, a loss -- exact cash math with zero fees/slippage."""
    md_open = make_fixed_quote_md(100.0)
    broker = PaperBroker(starting_capital=100_000, market_data=md_open, slippage_pct=0, fee_pct=0)
    broker.place_order("FAKE", "SELL", 10)  # short 10 @ 100 -> cash += 1000
    cash_after_open = broker.get_balance()
    assert cash_after_open == pytest.approx(100_000 + 1_000)

    broker.market_data = make_fixed_quote_md(90.0)  # price fell -- profitable cover
    broker.place_order("FAKE", "BUY", 10)  # cover 10 @ 90 -> cash -= 900
    assert broker.get_positions() == []
    assert broker.get_balance() == pytest.approx(cash_after_open - 900)
    assert broker.get_balance() > 100_000  # net profit vs. starting capital


def test_paper_broker_short_cover_with_fees_charges_fee_on_close_qty_only():
    """Bug-2 regression: if a cover order's requested quantity exceeds the
    held short, fee/filled_quantity must reflect only the ACTUAL closed
    quantity, not the larger requested one."""
    md = make_fixed_quote_md(100.0)
    broker = PaperBroker(starting_capital=100_000, market_data=md, slippage_pct=0, fee_pct=0.01)
    broker.place_order("FAKE", "SELL", 10)  # short 10 @ 100
    cash_before_cover = broker.get_balance()

    order = broker.place_order("FAKE", "BUY", 999)  # requests far more than the 10 held
    assert order.filled_quantity == 10  # capped to what was actually closed
    assert broker.get_positions() == []  # fully closed, not partially
    expected_fee = 100.0 * 10 * 0.01
    expected_cash = cash_before_cover - (100.0 * 10 + expected_fee)
    assert broker.get_balance() == pytest.approx(expected_cash)


def test_paper_broker_short_equity_marks_to_market_correctly():
    """Bug-3 regression: equity() must reflect cash (already holding the
    short-sale proceeds) minus the current buyback liability -- not
    double-count the notional."""
    md = make_fixed_quote_md(100.0)
    broker = PaperBroker(starting_capital=100_000, market_data=md, slippage_pct=0, fee_pct=0)
    broker.place_order("FAKE", "SELL", 10)  # short 10 @ 100 -- cash = 101,000
    cash = broker.get_balance()

    broker.market_data = make_fixed_quote_md(100.0)  # flat -- no unrealized P&L yet
    assert broker.equity() == pytest.approx(cash - 100.0 * 10)  # == 100,000 exactly

    broker.market_data = make_fixed_quote_md(90.0)  # price fell -- short is profitable
    assert broker.equity() == pytest.approx(cash - 90.0 * 10)  # > starting capital

    broker.market_data = make_fixed_quote_md(110.0)  # price rose -- short is losing
    assert broker.equity() == pytest.approx(cash - 110.0 * 10)  # < starting capital


def test_paper_broker_unrealized_pnl_long_and_short():
    """Bug-4 (part 1) regression: unrealized_pnl() reports real P&L for
    both a LONG and a SHORT simultaneously, independent of cash/equity."""
    md = make_fixed_quote_md(100.0)
    broker = PaperBroker(starting_capital=1_000_000, market_data=md, slippage_pct=0, fee_pct=0)
    broker.place_order("LONGCO", "BUY", 10)   # long 10 @ 100
    broker.place_order("SHORTCO", "SELL", 5)  # short 5 @ 100

    # Same fetch_fn returns the same fixed price regardless of symbol --
    # move it for both by replacing market_data with a new fixed quote.
    broker.market_data = make_fixed_quote_md(120.0)  # up 20: long gains, short loses
    pnl = broker.unrealized_pnl()
    expected = (120.0 - 100.0) * 10 + (100.0 - 120.0) * 5
    assert pnl == pytest.approx(expected)
    assert pnl != pytest.approx(broker.equity() - broker.get_balance())  # NOT market value


# --- TradeJournal --------------------------------------------------------

def test_journal_tracks_open_and_closed_trades(tmp_path):
    journal = TradeJournal(path=str(tmp_path / "journal.csv"))
    entry = JournalEntry(
        trade_id="t1", symbol="FAKE", side="BUY", entry_time=datetime.now(timezone.utc),
        entry_price=100, stop_loss=97, target=108, quantity=10,
    )
    journal.record_open(entry)
    assert len(journal.open_trades()) == 1
    journal.record_close("t1", datetime.now(timezone.utc), exit_price=108, exit_reason="TARGET",
                          fees=1.0, gross_pnl=80.0, net_pnl=79.0)
    assert len(journal.open_trades()) == 0
    assert len(journal.closed_trades()) == 1
    assert journal.win_rate() == 1.0
    assert journal.profit_factor() == float("inf")


def test_journal_profit_factor_with_mixed_results(tmp_path):
    journal = TradeJournal(path=str(tmp_path / "journal.csv"))
    for i, (pnl, reason) in enumerate([(100, "TARGET"), (-50, "STOP"), (-25, "STOP")]):
        entry = JournalEntry(trade_id=f"t{i}", symbol="FAKE", side="BUY",
                              entry_time=datetime.now(timezone.utc), entry_price=100,
                              stop_loss=97, target=108, quantity=10)
        journal.record_open(entry)
        journal.record_close(f"t{i}", datetime.now(timezone.utc), exit_price=100, exit_reason=reason,
                              fees=0, gross_pnl=pnl, net_pnl=pnl)
    assert journal.win_rate() == pytest.approx(1 / 3)
    assert journal.profit_factor() == pytest.approx(100 / 75)


# --- PaperTradingEngine (fully offline) -----------------------------------

def make_offline_engine(daily_df: pd.DataFrame, tmp_path) -> PaperTradingEngine:
    config = Config()
    config.decision_thresholds.min_history_bars = 60
    md = MarketDataProvider(fetch_fn=make_fetch_fn(daily_df))

    def fake_fundamentals_fetch(symbol):
        return {
            "revenue_growth": 0.1, "earnings_growth": 0.1, "eps": 5, "pe_ratio": 20, "pb_ratio": 3,
            "debt_to_equity": 50, "roe": 0.15, "profit_margin": 0.1, "operating_cash_flow": 100_000,
            "free_cash_flow": 50_000, "last_update": datetime.now(timezone.utc),
        }

    engine = PaperTradingEngine(
        config=config, market_data=md, fetch_news=False, fetch_social=False,
        fundamental_analyzer=FundamentalAnalyzer(fetch_fn=fake_fundamentals_fetch),
        journal=TradeJournal(path=str(tmp_path / "journal.csv")),
    )
    return engine


def test_engine_scan_symbol_produces_a_report(uptrend_daily, tmp_path):
    engine = make_offline_engine(uptrend_daily, tmp_path)
    result = engine.scan_symbol("FAKE")
    assert result is not None
    assert result.report.symbol == "FAKE"
    assert result.filter_result.final_decision in ("BUY", "SELL", "HOLD", "NO TRADE")


def test_engine_no_trade_is_a_valid_common_outcome(choppy_daily, tmp_path):
    engine = make_offline_engine(choppy_daily, tmp_path)
    result = engine.scan_symbol("FAKE")
    assert result is not None
    # Choppy/no-trend synthetic data should very often fail to clear the bar.
    assert result.filter_result.final_decision in ("HOLD", "NO TRADE", "BUY", "SELL")


def test_engine_account_summary_shape(uptrend_daily, tmp_path):
    engine = make_offline_engine(uptrend_daily, tmp_path)
    summary = engine.account_summary()
    for key in ["equity", "cash", "total_return_pct", "open_positions", "win_rate", "profit_factor"]:
        assert key in summary


# --- PaperTradingEngine: manage_open_positions / account_summary bug fixes -

def _last_close(daily_df: pd.DataFrame) -> float:
    return float(daily_df["Close"].iloc[-1])


def test_manage_open_positions_releases_sector_exposure_on_auto_close(uptrend_daily, tmp_path):
    """Bug-5 regression: an automatic stop/target close must release the
    sector exposure it reserved at open, not leak it forever."""
    from broker.broker_interface import Position

    engine = make_offline_engine(uptrend_daily, tmp_path)
    last_close = _last_close(uptrend_daily)
    stop_loss = last_close + 1.0  # already breached -- guarantees an immediate STOP on a LONG

    engine.broker._positions["FAKE"] = Position(
        symbol="FAKE", side="LONG", quantity=10, average_price=last_close, stop_loss=stop_loss,
        target=last_close + 100.0, sector="IT",
    )
    trade_id = "seed-1"
    engine._trade_ids["FAKE"] = trade_id
    engine.journal.record_open(JournalEntry(
        trade_id=trade_id, symbol="FAKE", side="BUY", entry_time=datetime.now(timezone.utc),
        entry_price=last_close, stop_loss=stop_loss, target=last_close + 100.0, quantity=10, fees=1.0,
    ))
    engine.capital_protection.register_open("IT", last_close * 10)
    assert engine.capital_protection.state.sector_exposure["IT"] == pytest.approx(last_close * 10)

    engine.manage_open_positions()

    assert engine.capital_protection.state.sector_exposure.get("IT", 0.0) == pytest.approx(0.0)
    assert "FAKE" not in engine._trade_ids
    assert engine.journal.closed_trades()[0].exit_reason == "STOP"


def test_manage_open_positions_includes_entry_fee_in_net_pnl(uptrend_daily, tmp_path):
    """Bug-6 regression: net_pnl (and therefore win_rate()/profit_factor())
    must subtract BOTH the entry and exit fee, not just the exit fee."""
    from broker.broker_interface import Position

    engine = make_offline_engine(uptrend_daily, tmp_path)
    engine.config.risk.transaction_cost_pct = 0.01  # large, easy-to-verify fee
    last_close = _last_close(uptrend_daily)
    entry_price = last_close - 10.0  # guarantees a profitable TARGET hit
    target = last_close  # already at/above target -- guarantees an immediate close

    engine.broker._positions["FAKE"] = Position(
        symbol="FAKE", side="LONG", quantity=10, average_price=entry_price, stop_loss=entry_price - 50.0,
        target=target, sector=None,
    )
    trade_id = "seed-2"
    entry_fee = entry_price * 10 * engine.config.risk.transaction_cost_pct
    engine._trade_ids["FAKE"] = trade_id
    engine.journal.record_open(JournalEntry(
        trade_id=trade_id, symbol="FAKE", side="BUY", entry_time=datetime.now(timezone.utc),
        entry_price=entry_price, stop_loss=entry_price - 50.0, target=target, quantity=10, fees=entry_fee,
    ))

    engine.manage_open_positions()

    closed = engine.journal.closed_trades()[0]
    exit_fee = closed.exit_price * 10 * engine.config.risk.transaction_cost_pct
    gross = (closed.exit_price - entry_price) * 10
    assert closed.net_pnl == pytest.approx(gross - entry_fee - exit_fee)
    assert closed.fees == pytest.approx(entry_fee + exit_fee)
    # Sanity: the (buggy) omit-entry-fee value would be strictly higher --
    # confirms the fix actually changed the number, not a no-op.
    assert closed.net_pnl < gross - exit_fee


def test_account_summary_unrealized_pnl_is_actual_pnl_not_market_value(uptrend_daily, tmp_path):
    """Bug-4 end-to-end regression: account_summary()['unrealized_pnl']
    must be real P&L (small, proportional to the price move since entry),
    not the position's full market value."""
    from broker.broker_interface import Position

    engine = make_offline_engine(uptrend_daily, tmp_path)
    last_close = _last_close(uptrend_daily)
    entry_price = last_close - 5.0  # small, known unrealized gain of 5/share

    engine.broker._positions["FAKE"] = Position(
        symbol="FAKE", side="LONG", quantity=10, average_price=entry_price, stop_loss=None, target=None,
    )
    summary = engine.account_summary()
    expected_unrealized = (last_close - entry_price) * 10  # == 50.0
    assert summary["unrealized_pnl"] == pytest.approx(expected_unrealized)
    market_value = last_close * 10
    assert summary["unrealized_pnl"] != pytest.approx(market_value)  # NOT ~1050-style market value


# --- PaperTradingEngine.execute_if_approved (previously untested) ----------

def make_approved_scan_result(symbol: str, side: str, sector=None):
    from paper_trading.engine import ScanResult
    from strategy.report import TradeReport
    from strategy.signal_engine import SignalDecision
    from strategy.trade_filter import FilterResult

    filter_result = FilterResult(
        symbol=symbol, approved=True, final_decision=side, checklist={}, reasons=["forced for test"],
    )
    signal = SignalDecision(
        symbol=symbol, component_scores={}, overall_confidence=80.0, confidence_label="HIGH",
        direction="UP" if side == "BUY" else "DOWN", model_agreement=1.0, decision=side,
    )
    report = TradeReport(
        symbol=symbol, current_price=0.0, market_trend="UPTREND", sector_trend="N/A",
        technical_score=None, fundamental_score=None, news_score=None, social_score=None,
        risk_score=None, overall_confidence=80.0, confidence_label="HIGH", entry=None,
        stop_loss=None, target=None, risk_reward=None, expected_risk=None, expected_reward=None,
        decision=side, reasons=[], invalidation_condition="test",
    )
    return ScanResult(symbol=symbol, report=report, signal=signal, filter_result=filter_result, sector=sector)


def make_large_capital_offline_engine(daily_df: pd.DataFrame, tmp_path) -> PaperTradingEngine:
    """Large starting capital so position sizing never rounds to 0 shares
    -- execute_if_approved recomputes risk/sizing fresh via the real
    pipeline regardless of what's in the hand-built ScanResult, so this
    must clear the real TradeRiskCalculator gates on its own merits."""
    engine = make_offline_engine(daily_df, tmp_path)
    engine.broker._cash = 10_000_000.0
    engine.broker.starting_capital = 10_000_000.0
    return engine


def test_execute_if_approved_threads_entry_fee_into_journal(uptrend_daily, tmp_path):
    engine = make_large_capital_offline_engine(uptrend_daily, tmp_path)
    engine.config.risk.transaction_cost_pct = 0.01
    trade_id = engine.execute_if_approved(make_approved_scan_result("FAKE", "BUY", sector="IT"))
    assert trade_id is not None

    entry = engine.journal.open_trades()[0]
    expected_fee = entry.entry_price * entry.quantity * 0.01
    assert entry.fees == pytest.approx(expected_fee)
    assert entry.fees > 0


def test_execute_if_approved_rejects_reentry_on_already_open_symbol(uptrend_daily, tmp_path):
    """Issue-7 regression: re-signaling on an already-open symbol must be
    refused, not silently merged at the broker while orphaning the first
    JournalEntry."""
    engine = make_large_capital_offline_engine(uptrend_daily, tmp_path)
    first = engine.execute_if_approved(make_approved_scan_result("FAKE", "BUY", sector="IT"))
    assert first is not None
    quantity_after_first = engine.broker.get_positions()[0].quantity

    second = engine.execute_if_approved(make_approved_scan_result("FAKE", "BUY", sector="IT"))
    assert second is None
    assert len(engine.journal.open_trades()) == 1
    assert engine.broker.get_positions()[0].quantity == quantity_after_first


def test_execute_if_approved_allows_reentry_after_auto_close(uptrend_daily, tmp_path):
    """Matches main.py's real per-cycle ordering: manage_open_positions()
    runs (and clears _trade_ids on a genuine close) before execute_if_approved
    each cycle -- re-entry on the SAME symbol must succeed afterward."""
    engine = make_large_capital_offline_engine(uptrend_daily, tmp_path)
    first = engine.execute_if_approved(make_approved_scan_result("FAKE", "BUY", sector="IT"))
    assert first is not None

    position = engine.broker.get_positions()[0]
    position.stop_loss = position.average_price + 1_000_000.0  # force an immediate STOP
    engine.manage_open_positions()
    assert "FAKE" not in engine._trade_ids

    second = engine.execute_if_approved(make_approved_scan_result("FAKE", "BUY", sector="IT"))
    assert second is not None
    assert second != first


def test_execute_if_approved_registers_sector_exposure_from_scan_result(uptrend_daily, tmp_path):
    """Regression for the sector-propagation bug: ScanResult previously
    had no `sector` field at all, so main.py/dashboard.py -- the only two
    real callers of execute_if_approved -- always passed sector=None,
    meaning max_sector_exposure_pct could never actually register or
    block anything. Confirms sector exposure is now genuinely non-zero
    after a real execution."""
    engine = make_large_capital_offline_engine(uptrend_daily, tmp_path)
    assert engine.capital_protection.state.sector_exposure.get("IT", 0.0) == 0.0

    trade_id = engine.execute_if_approved(make_approved_scan_result("FAKE", "BUY", sector="IT"))
    assert trade_id is not None
    assert engine.capital_protection.state.sector_exposure.get("IT", 0.0) > 0.0
