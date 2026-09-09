"""
Streamlit dashboard.

Run with:
    streamlit run dashboard/dashboard.py

Shows market condition, top candidates, confidence, entry/stop/target/RR,
open positions, P&L, drawdown, daily risk usage, recent news/sentiment, and
trade history -- and ALWAYS renders the current mode (PAPER vs LIVE)
prominently at the top, per spec section 19 ("never hide the mode").
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from config.settings import load_config
from dashboard.common import candidate_row, compute_market_regime, summarize_data_quality, top_candidates
from paper_trading.engine import PaperTradingEngine
from paper_trading.scanner import UniverseScanner

st.set_page_config(page_title="AI Trading System Dashboard", layout="wide")

config = load_config()

if "engine" not in st.session_state:
    st.session_state.engine = PaperTradingEngine(config)
if "scanner" not in st.session_state:
    st.session_state.scanner = UniverseScanner(st.session_state.engine)

engine: PaperTradingEngine = st.session_state.engine
scanner: UniverseScanner = st.session_state.scanner

# --- Mode banner -----------------------------------------------------------
if config.system.live_trading_enabled:
    st.error("LIVE MODE — real orders would be placed if a live broker were connected.", icon="🔴")
else:
    st.success("PAPER MODE — all orders are simulated. No real money is at risk.", icon="🟢")

st.title("AI-Assisted Trading Analysis & Paper Trading")
st.caption(
    "Confidence scores are calibrated estimates from historical/out-of-sample testing, "
    "not guarantees of profit and not claims of near-certain prediction accuracy."
)

# --- Sidebar controls --------------------------------------------------------
st.sidebar.header("Scan Settings")
symbols_input = st.sidebar.text_area("Symbols (comma-separated)", value=", ".join(config.universe.symbols))
symbols = [s.strip() for s in symbols_input.split(",") if s.strip()]
run_scan = st.sidebar.button("Run Scan")
run_manage = st.sidebar.button("Check Open Positions (stop/target)")
emergency_stop = st.sidebar.button("🛑 EMERGENCY STOP", type="primary")

if emergency_stop:
    engine.capital_protection.emergency_stop("Manual emergency stop from dashboard.")
    st.sidebar.warning("Emergency stop engaged. No new trades will be opened.")

if "scan_results" not in st.session_state:
    st.session_state.scan_results = []
if "scan_cycle" not in st.session_state:
    st.session_state.scan_cycle = None

if run_scan and symbols:
    with st.spinner("Scanning universe..."):
        st.session_state.scan_cycle = scanner.scan(symbols=symbols, force_refresh=True)
        st.session_state.scan_results = st.session_state.scan_cycle.results

if run_manage:
    engine.manage_open_positions()
    engine.record_equity_snapshot()

scan_results = st.session_state.scan_results
symbols_requested = getattr(st.session_state.get("scan_cycle"), "symbols_requested", len(scan_results))

# --- Market regime / data quality --------------------------------------------
regime = compute_market_regime(engine, config)
dq = summarize_data_quality(scan_results, symbols_requested=symbols_requested)
st.caption(f"**Market Regime:** {regime}  |  **Data Quality:** {dq.render_text()}")

# --- Account summary ---------------------------------------------------------
summary = engine.account_summary()
cols = st.columns(8)
cols[0].metric("Equity", f"{summary['equity']:.2f}")
cols[1].metric("Cash", f"{summary['cash']:.2f}")
cols[2].metric("Total Return", f"{summary['total_return_pct']:.2%}")
cols[3].metric("Drawdown (cur/max)", f"{summary['current_drawdown_pct']:.1%} / {summary['max_drawdown_pct']:.1%}")
cols[4].metric("Win Rate", f"{summary['win_rate']:.1%}")
cols[5].metric("Profit Factor", f"{summary['profit_factor']:.2f}")
cols[6].metric("Sharpe (session)", f"{summary['sharpe_ratio_session']:.2f}")
cols[7].metric("Closed Trades", summary["closed_trades"])

st.progress(
    min(summary["daily_risk_used_pct"], 1.0),
    text=f"Daily risk budget used: {summary['daily_risk_used_pct']:.0%} of max daily loss limit",
)

if summary["trading_halted"]:
    st.warning(f"Trading halted: {summary['halt_reason']}")

# --- Candidates ---------------------------------------------------------
st.subheader("Top Candidates")
if scan_results:
    rows = []
    for r in top_candidates(scan_results, top_n=10):
        row = candidate_row(r)
        rows.append({
            "Symbol": row["symbol"], "Price": row["price"], "Decision": row["decision"],
            "Confidence": f"{row['confidence']:.1f} ({row['confidence_label']})",
            "Technical": row["technical_score"], "Fundamental": row["fundamental_score"],
            "News": row["news_score"], "Social": row["social_score"], "ML": row["ml_score"],
            "Risk": row["risk_score"],
            "Entry": row["entry"], "Stop": row["stop_loss"], "Target": row["target"],
            "R:R": row["risk_reward"], "EV/share": row["expected_value_per_share"],
            "Data Quality": row["data_quality_status"],
            "Top Reason": row["reasons"][0] if row["reasons"] else "-",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    selected_symbol = st.selectbox("View full report for:", [r.symbol for r in scan_results])
    selected = next(r for r in scan_results if r.symbol == selected_symbol)
    st.text(selected.report.render_text())

    can_execute = selected.filter_result.approved and selected.filter_result.final_decision in ("BUY", "SELL")
    if st.button(f"Execute paper trade for {selected_symbol}", disabled=not can_execute):
        trade_id = engine.execute_if_approved(selected)
        if trade_id:
            st.success(f"Paper order placed for {selected_symbol}.")
        else:
            st.error("Order could not be executed (risk/account checks failed at execution time).")
else:
    st.info("Run a scan from the sidebar to see candidates. NO TRADE is the expected default outcome for most symbols, most of the time.")

# --- Open positions ---------------------------------------------------------
st.subheader("Open Positions")
positions = engine.broker.get_positions()
if positions:
    st.dataframe(pd.DataFrame([{
        "Symbol": p.symbol, "Side": p.side, "Qty": p.quantity, "Avg Price": p.average_price,
        "Stop": p.stop_loss, "Target": p.target,
    } for p in positions]), use_container_width=True)
else:
    st.write("No open positions.")

# --- Trade history ---------------------------------------------------------
st.subheader("Trade History (Journal)")
closed = engine.journal.closed_trades()
if closed:
    st.dataframe(pd.DataFrame([{
        "Symbol": e.symbol, "Side": e.side, "Entry": e.entry_price, "Exit": e.exit_price,
        "Exit Reason": e.exit_reason, "Net P&L": e.net_pnl, "Confidence at Entry": e.confidence_at_entry,
    } for e in closed]), use_container_width=True)
else:
    st.write("No closed trades yet.")

st.caption(
    "This dashboard is for analysis and paper trading only. See README.md section "
    "'Real-Money Readiness Checklist' before ever considering live trading."
)
