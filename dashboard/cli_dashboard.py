"""
Terminal dashboard (no extra services required).

Renders the same information the Streamlit dashboard shows, using `rich`
tables, directly in the console. Useful for headless environments, CI, and
quick checks. ALWAYS shows PAPER MODE / LIVE MODE prominently -- this must
never be ambiguous to whoever is reading it.

Spec section 18: shows Market Regime, Data Quality, Top Candidates (with
per-candidate Technical/Fundamental/News/Social/ML/Risk score, Entry/Stop/
Target/R:R/Expected Value, Decision, and Decision Reasons), and the paper
account's Equity/P&L/Drawdown/Win Rate/Profit Factor/Sharpe/Number of
Trades/Daily Risk Used.
"""

from __future__ import annotations

from typing import List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config.settings import Config
from dashboard.common import candidate_row, compute_market_regime, fmt_num, summarize_data_quality, top_candidates
from paper_trading.engine import PaperTradingEngine, ScanResult


def render_cli_dashboard(
    engine: PaperTradingEngine, scan_results: List[ScanResult], config: Config,
    symbols_requested: Optional[int] = None, top_n: int = 10,
) -> None:
    # Explicit width: with this many columns across the account/candidates
    # tables, relying on ambient terminal-width detection (which defaults to
    # something narrow when stdout isn't a real TTY, e.g. under a CI runner
    # or test capture) wraps every cell into an unreadable single-character-
    # per-line mess. A fixed wide width keeps this legible everywhere; a
    # real wide terminal still displays it fine.
    console = Console(width=220)

    mode = "LIVE MODE" if config.system.live_trading_enabled else "PAPER MODE"
    mode_style = "bold red" if config.system.live_trading_enabled else "bold green"
    console.print(Panel(f"[{mode_style}]{mode}[/{mode_style}]", title="Trading Mode", expand=False))

    regime = compute_market_regime(engine, config)
    dq = summarize_data_quality(scan_results, symbols_requested=symbols_requested if symbols_requested is not None else len(scan_results))
    console.print(f"[bold]Market Regime:[/bold] {regime}    [bold]Data Quality:[/bold] {dq.render_text()}")

    summary = engine.account_summary()
    acct_table = Table(title="Paper Account Summary")
    for col in ["Equity", "Cash", "Total Return", "Drawdown (cur/max)", "Win Rate", "Profit Factor",
                "Sharpe (session)", "Closed Trades", "Daily Risk Used", "Halted?"]:
        acct_table.add_column(col)
    acct_table.add_row(
        f"{summary['equity']:.2f}", f"{summary['cash']:.2f}", f"{summary['total_return_pct']:.2%}",
        f"{summary['current_drawdown_pct']:.2%} / {summary['max_drawdown_pct']:.2%}",
        f"{summary['win_rate']:.1%}", f"{summary['profit_factor']:.2f}",
        f"{summary['sharpe_ratio_session']:.2f}", str(summary["closed_trades"]),
        f"{summary['daily_risk_used_pct']:.0%}",
        "YES - " + summary["halt_reason"] if summary["trading_halted"] else "No",
    )
    console.print(acct_table)

    candidates_table = Table(title=f"Top Candidates (top {top_n} by confidence)")
    for col in ["Symbol", "Price", "Decision", "Confidence", "Tech", "Fund", "News", "Social", "ML", "Risk",
                "Entry", "Stop", "Target", "R:R", "EV/share", "Data Q", "Top Reason"]:
        candidates_table.add_column(col)
    for r in top_candidates(scan_results, top_n=top_n):
        row = candidate_row(r)
        candidates_table.add_row(
            row["symbol"], fmt_num(row["price"]), str(row["decision"]),
            f"{row['confidence']:.1f} ({row['confidence_label']})",
            fmt_num(row["technical_score"], ".0f"), fmt_num(row["fundamental_score"], ".0f"),
            fmt_num(row["news_score"], ".0f"), fmt_num(row["social_score"], ".0f"),
            fmt_num(row["ml_score"], ".0f"), fmt_num(row["risk_score"], ".0f"),
            fmt_num(row["entry"]), fmt_num(row["stop_loss"]), fmt_num(row["target"]),
            fmt_num(row["risk_reward"]), fmt_num(row["expected_value_per_share"], ".3f"),
            row["data_quality_status"], (row["reasons"][0] if row["reasons"] else "-"),
        )
    console.print(candidates_table)

    positions = engine.broker.get_positions()
    pos_table = Table(title="Open Positions")
    for col in ["Symbol", "Side", "Qty", "Avg Price", "Stop", "Target", "Last Quote"]:
        pos_table.add_column(col)
    for p in positions:
        try:
            quote = engine.broker.get_quote(p.symbol)
        except Exception:
            quote = float("nan")
        pos_table.add_row(p.symbol, p.side, str(p.quantity), f"{p.average_price:.2f}",
                           f"{p.stop_loss:.2f}" if p.stop_loss else "-",
                           f"{p.target:.2f}" if p.target else "-", f"{quote:.2f}")
    console.print(pos_table)

    journal_table = Table(title="Recent Trade History")
    for col in ["Symbol", "Side", "Entry", "Exit", "Exit Reason", "Net P&L"]:
        journal_table.add_column(col)
    for e in engine.journal.closed_trades()[-10:]:
        journal_table.add_row(
            e.symbol, e.side, f"{e.entry_price:.2f}",
            f"{e.exit_price:.2f}" if e.exit_price else "-",
            e.exit_reason or "-", f"{e.net_pnl:.2f}" if e.net_pnl is not None else "-",
        )
    console.print(journal_table)

    console.print(
        "[dim]Disclaimer: scores and confidence levels are calibrated estimates, "
        "not guarantees of profit or of near-certain prediction accuracy. ML score is only "
        "shown when this run was started with --ml; N/A means either no --ml flag was used, "
        "or the model had insufficient history/didn't clear its AUC gate for this symbol.[/dim]"
    )
