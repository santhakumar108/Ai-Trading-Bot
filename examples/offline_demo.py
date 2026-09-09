#!/usr/bin/env python3
"""
Fully offline, synthetic-data demo.

Useful for:
  * Verifying the whole pipeline works end-to-end without any network access
    (this sandbox environment's yfinance/Google News/Reddit calls are
    blocked by default network policy in some hosting environments -- this
    script never needs them).
  * A quick, reproducible walkthrough of what a scan/paper-trade cycle looks
    like, including a realistic mix of BUY, HOLD, and NO TRADE outcomes.

Run with:
    python examples/offline_demo.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config.settings import Config
from data.market_data import MarketDataProvider
from dashboard.cli_dashboard import render_cli_dashboard
from dashboard.html_report import render_html_report
from fundamentals.fundamental_analysis import FundamentalAnalyzer
from paper_trading.engine import PaperTradingEngine
from paper_trading.journal import TradeJournal


def synthetic_ohlcv(n, drift, volatility, seed, start=100.0):
    rng = np.random.default_rng(seed)
    returns = rng.normal(drift, volatility, n)
    close = start * np.cumprod(1 + returns)
    open_ = np.empty(n)
    open_[0] = start
    open_[1:] = close[:-1]
    rng2 = np.random.default_rng(seed + 1)
    spread = np.abs(rng2.normal(volatility * 0.6, volatility * 0.3, n)) * close
    high = np.maximum(open_, close) + spread * 0.5
    low = np.clip(np.minimum(open_, close) - spread * 0.5, 0.01, None)
    volume = rng.integers(200_000, 3_000_000, n).astype(float)
    dates = pd.bdate_range("2023-01-02", periods=n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=dates)


# A small, deliberately mixed universe: one clean uptrend (should be able to
# produce a BUY if everything aligns), one downtrend, one choppy/sideways
# series, and one illiquid one -- so the demo shows NO TRADE for the right
# reasons, not just for one reason repeated four times.
SYMBOL_DATA = {
    "DEMO_UP": synthetic_ohlcv(300, drift=0.0035, volatility=0.010, seed=10),
    "DEMO_DOWN": synthetic_ohlcv(300, drift=-0.0030, volatility=0.010, seed=20),
    "DEMO_CHOPPY": synthetic_ohlcv(300, drift=0.0000, volatility=0.009, seed=30),
    "DEMO_ILLIQUID": synthetic_ohlcv(300, drift=0.0035, volatility=0.010, seed=10) .assign(Volume=lambda d: d["Volume"] * 0.01),
}
INDEX_DATA = synthetic_ohlcv(300, drift=0.0008, volatility=0.008, seed=99)


def fetch_fn(symbol, period, interval):
    if symbol == "^GSPC":
        return INDEX_DATA.copy()
    return SYMBOL_DATA[symbol].copy()


def fake_fundamentals(symbol):
    good = symbol in ("DEMO_UP", "DEMO_ILLIQUID")
    if good:
        return {
            "revenue_growth": 0.16, "earnings_growth": 0.20, "eps": 8.0, "pe_ratio": 19, "pb_ratio": 3.2,
            "debt_to_equity": 45, "roe": 0.22, "profit_margin": 0.17, "operating_cash_flow": 2_000_000,
            "free_cash_flow": 1_200_000, "last_update": datetime.now(timezone.utc),
        }
    return {
        "revenue_growth": -0.05, "earnings_growth": -0.10, "eps": -0.5, "pe_ratio": 70, "pb_ratio": 9,
        "debt_to_equity": 220, "roe": -0.05, "profit_margin": -0.02, "operating_cash_flow": -200_000,
        "free_cash_flow": -400_000, "last_update": datetime.now(timezone.utc),
    }


def main():
    config = Config()
    config.decision_thresholds.min_history_bars = 60
    config.universe.symbols = list(SYMBOL_DATA.keys())
    config.universe.index_symbol = "^GSPC"

    md = MarketDataProvider(fetch_fn=fetch_fn)
    engine = PaperTradingEngine(
        config=config, market_data=md, fetch_news=False, fetch_social=False,
        fundamental_analyzer=FundamentalAnalyzer(fetch_fn=fake_fundamentals),
        journal=TradeJournal(path="logs/offline_demo_journal.csv"),
    )

    print(f"Mode: {'LIVE' if config.system.live_trading_enabled else 'PAPER'}\n")
    results = engine.scan_universe(list(SYMBOL_DATA.keys()))

    for r in results:
        print(r.report.render_text())
        print("=" * 80)

    executed = []
    for r in results:
        trade_id = engine.execute_if_approved(r)
        if trade_id:
            executed.append(r.symbol)
    print(f"\nExecuted paper trades for: {executed if executed else '(none -- NO TRADE was the outcome for all symbols)'}")

    engine.manage_open_positions()

    print("\nAccount summary:", engine.account_summary())

    render_cli_dashboard(engine, results, config)
    html_path = render_html_report(engine, results, config, "logs/offline_demo_dashboard.html")
    print(f"\nWrote HTML dashboard snapshot to {html_path}")


if __name__ == "__main__":
    main()
