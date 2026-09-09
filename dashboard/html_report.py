"""
Static HTML snapshot report.

Generates a single self-contained HTML file with the same information as
the live dashboards (account summary, candidates, open positions, trade
history), for sharing a point-in-time snapshot without running Streamlit.
Not a live/auto-refreshing page -- regenerate it whenever you want a fresh
snapshot (see main.py's `dashboard --html` option).

Spec section 18: shows Market Regime, Data Quality, Top Candidates (with
per-candidate Technical/Fundamental/News/Social/ML/Risk score, Entry/Stop/
Target/R:R/Expected Value, Decision, and Decision Reasons), and the paper
account's Equity/P&L/Drawdown/Win Rate/Profit Factor/Sharpe/Number of
Trades/Daily Risk Used.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from config.settings import Config
from dashboard.common import candidate_row, compute_market_regime, fmt_num, summarize_data_quality, top_candidates
from paper_trading.engine import PaperTradingEngine, ScanResult

_CSS = """
body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; background:#0b0f14; color:#e6edf3; margin:0; padding:24px; }
h1, h2 { font-weight: 600; }
.badge { display:inline-block; padding:6px 14px; border-radius:6px; font-weight:700; letter-spacing:0.5px; }
.paper { background:#123d24; color:#3fb950; border:1px solid #3fb950; }
.live { background:#3d1212; color:#f85149; border:1px solid #f85149; }
table { border-collapse: collapse; width:100%; margin-bottom:28px; background:#11161c; }
th, td { border:1px solid #21262d; padding:8px 10px; font-size:13px; text-align:left; }
th { background:#161b22; color:#8b949e; text-transform:uppercase; font-size:11px; }
.decision-BUY { color:#3fb950; font-weight:700; }
.decision-SELL { color:#f85149; font-weight:700; }
.decision-HOLD { color:#d29922; font-weight:700; }
.decision-NOTRADE, .decision-NO { color:#8b949e; font-weight:700; }
.section { margin-bottom: 36px; }
.status-line { margin: 8px 0 24px 0; color:#c9d1d9; }
.disclaimer { color:#8b949e; font-size:12px; margin-top:40px; border-top:1px solid #21262d; padding-top:12px; }
"""


def _decision_class(decision: str) -> str:
    return "decision-" + decision.replace(" ", "").replace("_", "")


def render_html_report(
    engine: PaperTradingEngine, scan_results: List[ScanResult], config: Config, out_path: str,
    symbols_requested: Optional[int] = None, top_n: int = 10,
) -> str:
    mode_label = "LIVE MODE" if config.system.live_trading_enabled else "PAPER MODE"
    mode_class = "live" if config.system.live_trading_enabled else "paper"
    summary = engine.account_summary()
    regime = compute_market_regime(engine, config)
    dq = summarize_data_quality(
        scan_results, symbols_requested=symbols_requested if symbols_requested is not None else len(scan_results),
    )

    def _candidate_row(r) -> str:
        row = candidate_row(r)
        return (
            f"<tr><td>{row['symbol']}</td><td>{fmt_num(row['price'])}</td>"
            f"<td class='{_decision_class(str(row['decision']))}'>{row['decision']}</td>"
            f"<td>{row['confidence']:.1f} ({row['confidence_label']})</td>"
            f"<td>{fmt_num(row['technical_score'], '.0f')}</td>"
            f"<td>{fmt_num(row['fundamental_score'], '.0f')}</td>"
            f"<td>{fmt_num(row['news_score'], '.0f')}</td>"
            f"<td>{fmt_num(row['social_score'], '.0f')}</td>"
            f"<td>{fmt_num(row['ml_score'], '.0f')}</td>"
            f"<td>{fmt_num(row['risk_score'], '.0f')}</td>"
            f"<td>{fmt_num(row['entry'])}</td>"
            f"<td>{fmt_num(row['stop_loss'])}</td>"
            f"<td>{fmt_num(row['target'])}</td>"
            f"<td>{fmt_num(row['risk_reward'])}</td>"
            f"<td>{fmt_num(row['expected_value_per_share'], '.3f')}</td>"
            f"<td>{row['data_quality_status']}</td>"
            f"<td>{'; '.join(row['reasons']) if row['reasons'] else '-'}</td></tr>"
        )

    candidate_rows = "".join(_candidate_row(r) for r in top_candidates(scan_results, top_n=top_n))

    position_rows = "".join(
        f"<tr><td>{p.symbol}</td><td>{p.side}</td><td>{p.quantity}</td>"
        f"<td>{p.average_price:.2f}</td><td>{(p.stop_loss or 0):.2f}</td>"
        f"<td>{(p.target or 0):.2f}</td></tr>"
        for p in engine.broker.get_positions()
    )

    trade_rows = "".join(
        f"<tr><td>{e.symbol}</td><td>{e.side}</td><td>{e.entry_price:.2f}</td>"
        f"<td>{(e.exit_price or 0):.2f}</td><td>{e.exit_reason or '-'}</td>"
        f"<td>{(e.net_pnl if e.net_pnl is not None else 0):.2f}</td></tr>"
        for e in engine.journal.closed_trades()[-25:]
    )

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Trading System Dashboard</title>
<style>{_CSS}</style></head>
<body>
<h1>AI-Assisted Trading Analysis &mdash; Dashboard Snapshot</h1>
<p><span class="badge {mode_class}">{mode_label}</span> &nbsp; Generated: {datetime.utcnow().isoformat()}Z</p>
<p class="status-line"><strong>Market Regime:</strong> {regime} &nbsp;&nbsp; <strong>Data Quality:</strong> {dq.render_text()}</p>

<div class="section">
<h2>Paper Account Summary</h2>
<table>
<tr><th>Equity</th><th>Cash</th><th>Total Return</th><th>Drawdown (cur/max)</th><th>Win Rate</th>
<th>Profit Factor</th><th>Sharpe (session)</th><th>Closed Trades</th><th>Daily Risk Used</th><th>Halted?</th></tr>
<tr>
<td>{summary['equity']:.2f}</td><td>{summary['cash']:.2f}</td><td>{summary['total_return_pct']:.2%}</td>
<td>{summary['current_drawdown_pct']:.2%} / {summary['max_drawdown_pct']:.2%}</td>
<td>{summary['win_rate']:.1%}</td><td>{summary['profit_factor']:.2f}</td>
<td>{summary['sharpe_ratio_session']:.2f}</td><td>{summary['closed_trades']}</td>
<td>{summary['daily_risk_used_pct']:.0%}</td>
<td>{'YES - ' + summary['halt_reason'] if summary['trading_halted'] else 'No'}</td>
</tr>
</table>
</div>

<div class="section">
<h2>Top Candidates (top {top_n} by confidence)</h2>
<table>
<tr><th>Symbol</th><th>Price</th><th>Decision</th><th>Confidence</th><th>Tech</th><th>Fund</th><th>News</th>
<th>Social</th><th>ML</th><th>Risk</th><th>Entry</th><th>Stop</th><th>Target</th><th>R:R</th>
<th>EV/share</th><th>Data Q</th><th>Decision Reasons</th></tr>
{candidate_rows}
</table>
</div>

<div class="section">
<h2>Open Positions</h2>
<table>
<tr><th>Symbol</th><th>Side</th><th>Qty</th><th>Avg Price</th><th>Stop</th><th>Target</th></tr>
{position_rows}
</table>
</div>

<div class="section">
<h2>Recent Trade History</h2>
<table>
<tr><th>Symbol</th><th>Side</th><th>Entry</th><th>Exit</th><th>Exit Reason</th><th>Net P&amp;L</th></tr>
{trade_rows}
</table>
</div>

<p class="disclaimer">This system does not guarantee profit and does not claim near-certain
prediction accuracy. Confidence scores are calibrated estimates based on historical and
out-of-sample backtesting, not statements of probability of profit on any individual trade.
Paper trading results do not guarantee live-market results, which are affected by real
liquidity, slippage, and execution conditions not fully captured here. ML score is only shown
when this run was started with --ml; N/A means either no --ml flag was used, or the model had
insufficient history/didn't clear its AUC gate for this symbol.</p>
</body></html>"""

    with open(out_path, "w") as f:
        f.write(html)
    return out_path
