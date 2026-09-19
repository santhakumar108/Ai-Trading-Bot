"""
Customer-facing HTML dashboard generator.

Renders a simplified, non-technical, consumer-app-style dashboard (no
technical/fundamental/ML scores, no gate-by-gate reasons -- just balance,
holdings, P&L, trade counts, and plain-language explanations) with REAL
numbers read from the given PaperTradingEngine's current state. Engine
construction reconstructs cash/positions/capital-protection state from
the journal's persisted history (PaperTradingEngine._restore_state_from_journal),
so this reflects the account's real cumulative position across every
past run, not just the current process.

Never fabricates data: a panel with zero real rows shows an honest empty
state plus a clearly-labeled SAMPLE block illustrating the format. The
portfolio-value line is real (starting capital -> current equity, two
real points) rather than a fabricated intermediate path -- no invented
day-to-day wiggle, since no persisted daily equity-curve exists yet.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List, Optional

from broker.broker_interface import Position
from config.nse_company_names import NSE_COMPANY_NAMES
from config.settings import Config
from paper_trading.engine import PaperTradingEngine
from paper_trading.journal import JournalEntry

IST_OFFSET = timedelta(hours=5, minutes=30)


def _to_ist(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc) + IST_OFFSET


def _inr(amount: float) -> str:
    n = int(round(abs(amount)))
    s = str(n)
    if len(s) <= 3:
        grouped = s
    else:
        last3 = s[-3:]
        rest = s[:-3]
        parts: List[str] = []
        while len(rest) > 2:
            parts.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            parts.insert(0, rest)
        grouped = ",".join(parts) + "," + last3
    return f"&#8377;{grouped}"


def _signed_inr(amount: float) -> str:
    sign = "+" if amount >= 0 else "-"
    return f"{sign}{_inr(amount)}"


def _pct(value: float) -> str:
    arrow = "&#9650;" if value >= 0 else "&#9660;"
    return f"{arrow} {abs(value):.1f}%"


def _inr_abbrev(amount: float) -> str:
    """Compact Indian-style abbreviation for tight spaces (e.g. the
    floating chart badge) -- Crore (&#8377;1.25Cr) / Lakh (&#8377;10.08L)
    / plain rupees below a lakh."""
    n = abs(amount)
    sign = "-" if amount < 0 else ""
    if n >= 1_00_00_000:
        return f"{sign}&#8377;{n / 1_00_00_000:.2f}Cr"
    if n >= 1_00_000:
        return f"{sign}&#8377;{n / 1_00_000:.2f}L"
    return _inr(amount)


def _short_symbol(symbol: str) -> str:
    return symbol[:-3] if symbol.endswith(".NS") else symbol


def _company_name(symbol: str) -> str:
    return NSE_COMPANY_NAMES.get(symbol, _short_symbol(symbol))


def _avatar(symbol: str) -> str:
    return _short_symbol(symbol)[:3].upper()


def _relative_day_label(d, today) -> str:
    delta = (today - d).days
    if delta == 0:
        return "Today"
    if delta == 1:
        return "Yesterday"
    if 1 < delta < 7:
        return f"{delta} days ago"
    return d.strftime("%d %b")


def _next_check_label(now_ist: datetime) -> str:
    """Plain-language description of when the bot will next scan the
    market -- the real schedule is weekdays after market close (~4-6pm
    IST); this never claims a more precise time than that."""
    weekday = now_ist.weekday()  # 0=Mon .. 6=Sun
    if weekday < 5 and now_ist.hour < 16:
        return "Today after market close"
    days_ahead = 1
    candidate = now_ist + timedelta(days=days_ahead)
    while candidate.weekday() >= 5:
        days_ahead += 1
        candidate = now_ist + timedelta(days=days_ahead)
    if days_ahead == 1:
        return "Tomorrow after market close"
    return candidate.strftime("%A") + " after market close"


def _period_pnl_summary(closed: List[JournalEntry], now_ist: datetime) -> dict:
    """Realized P&L (closed trades only) grouped by the period their EXIT
    fell in, IST-aware. Unrealized P&L from still-open positions belongs
    to no specific closing day, so it's deliberately excluded here (it
    stays part of "Total Profit/Loss", which already blends both)."""
    today = now_ist.date()
    week_start = today - timedelta(days=today.weekday())
    month_start = today.replace(day=1)

    def _sum_and_count(predicate):
        matched = [e for e in closed if e.exit_time and predicate(_to_ist(e.exit_time).date())]
        return sum(e.net_pnl or 0.0 for e in matched), len(matched)

    return {
        "Today": _sum_and_count(lambda d: d == today),
        "This Week": _sum_and_count(lambda d: d >= week_start),
        "This Month": _sum_and_count(lambda d: d >= month_start),
        "All Time": (sum(e.net_pnl or 0.0 for e in closed), len(closed)),
    }


def _result_card_html(label: str, pnl: float, sub_text: str) -> str:
    cls = "neutral" if pnl == 0 else ("profit" if pnl > 0 else "loss")
    icon = "&#128197;" if label == "Today" else ("&#128202;" if label == "This Week" else "&#127942;")
    return (
        f'<div class="result-card {cls}">'
        f'<div class="result-icon">{icon}</div>'
        f'<p class="result-label">{label}</p>'
        f'<p class="result-value {cls} tabular">{_signed_inr(pnl)}</p>'
        f'<p class="result-sub">{sub_text}</p>'
        f'</div>'
    )


def _exit_plan_html(stop_loss: Optional[float], target: Optional[float]) -> str:
    if stop_loss is None or target is None:
        return ""
    return (
        f'<div class="exit-plan">'
        f'<div class="exit-card stop">'
        f'<div class="exit-icon">&#128737;</div>'
        f'<div><p class="exit-label">Safety price</p>'
        f'<p class="exit-value tabular">{_inr(stop_loss)}</p>'
        f'<p class="exit-caption">If the price falls to this level, the bot will sell to limit the loss.</p></div>'
        f'</div>'
        f'<div class="exit-card target">'
        f'<div class="exit-icon">&#127919;</div>'
        f'<div><p class="exit-label">Profit target</p>'
        f'<p class="exit-value tabular">{_inr(target)}</p>'
        f'<p class="exit-caption">If the price reaches this level, the bot may sell and take profit.</p></div>'
        f'</div>'
        f'</div>'
    )


def _holding_card_html(pos: Position, current_price: Optional[float]) -> str:
    price = current_price if current_price is not None else pos.average_price
    market_value = price * pos.quantity
    if pos.side == "LONG":
        pnl = (price - pos.average_price) * pos.quantity
        why = "The bot expects this stock may continue rising."
    else:
        pnl = (pos.average_price - price) * pos.quantity
        why = "The bot expects this stock may continue falling."
    denom = pos.average_price * pos.quantity
    pnl_pct = (pnl / denom * 100) if denom else 0.0
    pnl_cls = "profit" if pnl >= 0 else "loss"
    return (
        f'<div class="holding-card">'
        f'<div class="holding-top">'
        f'<div class="stock-avatar">{_avatar(pos.symbol)}</div>'
        f'<div class="holding-name">'
        f'<p class="h-symbol">{_short_symbol(pos.symbol)}</p>'
        f'<p class="h-company">{_company_name(pos.symbol)}</p>'
        f'</div>'
        f'<span class="holding-tag">Holding</span>'
        f'</div>'
        f'<div class="holding-grid">'
        f'<div><p class="hg-label">Shares</p><p class="hg-value tabular">{pos.quantity}</p></div>'
        f'<div><p class="hg-label">Bought at</p><p class="hg-value tabular">{_inr(pos.average_price)}</p></div>'
        f'<div><p class="hg-label">Current value</p><p class="hg-value tabular">{_inr(market_value)}</p></div>'
        f'<div><p class="hg-label">Profit / Loss</p><p class="hg-value {pnl_cls} tabular">{_signed_inr(pnl)} ({pnl_pct:+.1f}%)</p></div>'
        f'</div>'
        f'<div class="why-holding"><span class="why-icon">&#128200;</span>'
        f'<div><p class="why-title">Why is the bot holding this?</p><p class="why-body">{why}</p></div></div>'
        f'{_exit_plan_html(pos.stop_loss, pos.target)}'
        f'</div>'
    )


def _activity_row_html(e: JournalEntry, now_ist: datetime) -> str:
    is_close = e.exit_time is not None
    when_ist = _to_ist(e.exit_time if is_close else e.entry_time)
    day_num = when_ist.strftime("%d")
    month_short = when_ist.strftime("%b").upper()
    day_label = _relative_day_label(when_ist.date(), now_ist.date())
    time_label = when_ist.strftime("%I:%M %p").lstrip("0")

    if is_close:
        verb = "Sold" if e.side.upper() == "BUY" else "Bought back"
        badge_letter, badge_cls = "S", "sell"
        pnl = e.net_pnl or 0.0
        denom = e.entry_price * e.quantity
        pnl_pct = (pnl / denom * 100) if denom else 0.0
        status_cls = "profit" if pnl >= 0 else "loss"
        status_html = f'<span class="status-pill {status_cls}">{_signed_inr(pnl)} ({pnl_pct:+.1f}%)</span>'
        price_txt = f"at {_inr(e.exit_price or 0.0)} each"
    else:
        verb = "Bought" if e.side.upper() == "BUY" else "Sold short"
        badge_letter, badge_cls = "B", "buy"
        status_html = '<span class="status-pill open">Still holding</span>'
        price_txt = f"at {_inr(e.entry_price)} each"

    return (
        f'<div class="activity-row">'
        f'<div class="date-badge"><span class="dn">{day_num}</span><span class="dm">{month_short}</span></div>'
        f'<div class="activity-badge {badge_cls}">{badge_letter}</div>'
        f'<div class="activity-main">'
        f'<p class="a-title">{verb} {_short_symbol(e.symbol)}</p>'
        f'<p class="a-sub">{day_label}, {time_label}</p>'
        f'</div>'
        f'<div class="activity-right">'
        f'<p class="a-qty tabular">{e.quantity} shares</p>'
        f'<p class="a-price tabular">{price_txt}</p>'
        f'{status_html}'
        f'</div>'
        f'</div>'
    )


_HOLDINGS_SAMPLE = """
<div class="sample-banner">
  <span class="sample-tag">Sample</span>
  <span class="sample-note">The exact format each holding will use</span>
</div>
<div class="sample-block">
  <div class="holding-card">
    <div class="holding-top">
      <div class="stock-avatar">REL</div>
      <div class="holding-name"><p class="h-symbol">RELIANCE</p><p class="h-company">Reliance Industries Limited</p></div>
      <span class="holding-tag">Holding</span>
    </div>
    <div class="holding-grid">
      <div><p class="hg-label">Shares</p><p class="hg-value tabular">40</p></div>
      <div><p class="hg-label">Bought at</p><p class="hg-value tabular">&#8377;1,250.00</p></div>
      <div><p class="hg-label">Current value</p><p class="hg-value tabular">&#8377;51,200</p></div>
      <div><p class="hg-label">Profit / Loss</p><p class="hg-value profit tabular">+&#8377;2,400 (+4.9%)</p></div>
    </div>
    <div class="why-holding"><span class="why-icon">&#128200;</span>
      <div><p class="why-title">Why is the bot holding this?</p><p class="why-body">The bot expects this stock may continue rising.</p></div></div>
    <div class="exit-plan">
      <div class="exit-card stop"><div class="exit-icon">&#128737;</div><div><p class="exit-label">Safety price</p><p class="exit-value tabular">&#8377;1,190.00</p><p class="exit-caption">If the price falls to this level, the bot will sell to limit the loss.</p></div></div>
      <div class="exit-card target"><div class="exit-icon">&#127919;</div><div><p class="exit-label">Profit target</p><p class="exit-value tabular">&#8377;1,415.00</p><p class="exit-caption">If the price reaches this level, the bot may sell and take profit.</p></div></div>
    </div>
  </div>
</div>
"""

_ACTIVITY_SAMPLE = """
<div class="sample-banner">
  <span class="sample-tag">Sample</span>
  <span class="sample-note">What a real trade looks like</span>
</div>
<div class="sample-block">
  <div class="activity-row">
    <div class="date-badge"><span class="dn">12</span><span class="dm">SEP</span></div>
    <div class="activity-badge buy">B</div>
    <div class="activity-main"><p class="a-title">Bought RELIANCE</p><p class="a-sub">Today, 4:12 PM</p></div>
    <div class="activity-right"><p class="a-qty tabular">40 shares</p><p class="a-price tabular">at &#8377;1,250 each</p><span class="status-pill open">Still holding</span></div>
  </div>
  <div class="activity-row">
    <div class="date-badge"><span class="dn">11</span><span class="dm">SEP</span></div>
    <div class="activity-badge sell">S</div>
    <div class="activity-main"><p class="a-title">Sold INFY</p><p class="a-sub">Yesterday, 4:07 PM</p></div>
    <div class="activity-right"><p class="a-qty tabular">25 shares</p><p class="a-price tabular">at &#8377;1,480 each</p><span class="status-pill loss">-&#8377;875 (-2.4%)</span></div>
  </div>
</div>
"""


def render_customer_html(engine: PaperTradingEngine, config: Config) -> str:
    starting_capital = config.paper_trading.starting_capital
    equity = engine.broker.equity()
    change = equity - starting_capital
    change_pct = (change / starting_capital * 100) if starting_capital else 0.0

    positions = engine.broker.get_positions()
    closed = engine.journal.closed_trades()
    opened = engine.journal.open_trades()
    all_trades: List[JournalEntry] = sorted(
        closed + opened, key=lambda e: e.exit_time or e.entry_time, reverse=True,
    )

    now_ist = _to_ist(datetime.now(timezone.utc))
    today_ist = now_ist.date()

    def _touched_today(e: JournalEntry) -> bool:
        if _to_ist(e.entry_time).date() == today_ist:
            return True
        return bool(e.exit_time and _to_ist(e.exit_time).date() == today_ist)

    trades_today = sum(1 for e in (closed + opened) if _touched_today(e))
    total_trades = len(closed) + len(opened)
    total_gain_loss = sum(e.net_pnl or 0.0 for e in closed) + engine.broker.unrealized_pnl()

    invested_amount = 0.0
    if positions:
        rows = []
        for pos in positions:
            try:
                quote = engine.broker.get_quote(pos.symbol)
            except Exception:
                quote = None
            invested_amount += (quote if quote is not None else pos.average_price) * pos.quantity
            rows.append(_holding_card_html(pos, quote))
        holdings_section = f'<div class="real-block">{"".join(rows)}</div>'
    else:
        holdings_section = (
            '<div class="empty-card"><strong>0 holdings today.</strong> '
            'Your money is safely sitting as cash.</div>' + _HOLDINGS_SAMPLE
        )

    if all_trades:
        rows = [_activity_row_html(e, now_ist) for e in all_trades[:10]]
        activity_section = f'<div class="real-block">{"".join(rows)}</div>'
    else:
        activity_section = (
            '<div class="empty-card"><strong>0 trades today.</strong> '
            'New trades appear here the moment they happen.</div>' + _ACTIVITY_SAMPLE
        )

    period_summary = _period_pnl_summary(closed, now_ist)

    def _closed_sub(count: int, zero_text: str) -> str:
        if count == 0:
            return zero_text
        return f'{count} closed trade{"s" if count != 1 else ""}'

    today_pnl, today_n = period_summary["Today"]
    week_pnl, week_n = period_summary["This Week"]
    all_pnl, all_n = period_summary["All Time"]
    results_section = (
        _result_card_html("Today", today_pnl, _closed_sub(today_n, "No new trades"))
        + _result_card_html("This Week", week_pnl, _closed_sub(week_n, "No closed trades"))
        + _result_card_html(
            "Overall (Since Start)", all_pnl,
            f'{len(positions)} active position{"s" if len(positions) != 1 else ""}',
        )
    )

    universe_size = len(config.universe.symbols)
    as_of = now_ist.strftime("%a, %d %b &middot; %I:%M %p IST").replace(" 0", " ")

    if trades_today == 0:
        trade_status_title = "No new trade today"
        trade_status_body = "Your bot checked the market and didn't find a strong opportunity. That's a good thing!"
    else:
        trade_status_title = f"{trades_today} new trade{'s' if trades_today != 1 else ''} today"
        trade_status_body = "Your bot found a genuinely confident opportunity and acted on it."

    chart_svg = _portfolio_chart_svg(starting_capital, equity)

    html = _PAGE_TEMPLATE
    replacements = {
        "@@AS_OF@@": as_of,
        "@@UNIVERSE_SIZE@@": f"{universe_size:,}",
        "@@BALANCE@@": _inr(equity),
        "@@CHANGE_ARROW@@": "&#9650;" if change > 0 else ("&#9660;" if change < 0 else "&#8596;"),
        "@@CHANGE_CLASS@@": "profit" if change > 0 else ("loss" if change < 0 else "neutral"),
        "@@CHANGE_AMOUNT@@": _signed_inr(change),
        "@@CHANGE_PCT@@": f"{change_pct:+.1f}%",
        "@@STARTING_CAPITAL@@": _inr(starting_capital),
        "@@CHART_SVG@@": chart_svg,
        "@@CHART_BADGE@@": _inr_abbrev(equity),
        "@@AVAILABLE_CASH@@": _inr(engine.broker.get_balance()),
        "@@INVESTED_AMOUNT@@": _inr(invested_amount),
        "@@TOTAL_PL@@": _signed_inr(total_gain_loss),
        "@@TOTAL_PL_CLASS@@": "profit" if total_gain_loss >= 0 else "loss",
        "@@TRADE_STATUS_TITLE@@": trade_status_title,
        "@@TRADE_STATUS_BODY@@": trade_status_body,
        "@@NEXT_CHECK@@": _next_check_label(now_ist),
        "@@STOCK_COUNT@@": str(len(positions)),
        "@@STOCK_WORD@@": "stock" if len(positions) == 1 else "stocks",
        "@@HOLDINGS_SECTION@@": holdings_section,
        "@@ACTIVITY_SECTION@@": activity_section,
        "@@RESULTS_SECTION@@": results_section,
    }
    for token, value in replacements.items():
        html = html.replace(token, value)
    return html


def _portfolio_chart_svg(start_value: float, end_value: float) -> str:
    """Two real points only (starting capital -> current equity) -- no
    fabricated intermediate path, since no persisted daily equity-curve
    exists yet (see Phase 20's plan)."""
    y_start, y_end = 150.0, 40.0
    if end_value >= start_value:
        path_y_end = 40.0
    else:
        path_y_end = 150.0
    color = "var(--profit)" if end_value >= start_value else "var(--loss)"
    return (
        f'<svg viewBox="0 0 400 180" width="100%" height="140" role="img" aria-label="Portfolio value from start to now">'
        f'<line x1="0" y1="90" x2="400" y2="90" stroke="var(--border)" stroke-width="1" stroke-dasharray="4 4"></line>'
        f'<path d="M10,{y_start} L390,{path_y_end} L390,180 L10,180 Z" fill="{color}" opacity="0.10"></path>'
        f'<path d="M10,{y_start} L390,{path_y_end}" fill="none" stroke="{color}" stroke-width="3" stroke-linecap="round"></path>'
        f'<circle cx="390" cy="{path_y_end}" r="6" fill="{color}"></circle>'
        f'</svg>'
    )


_PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Santha's Trading Bot</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap">
<style>
  :root {
    --bg: #0A1512; --surface: #10201B; --surface-2: #152A23;
    --ink: #F1F5F3; --ink-soft: #9DB3AB; --ink-faint: #5F7A72;
    --border: #1E3830; --brand: #10B981; --brand-soft: #123B2E;
    --gold: #E3B341; --gold-soft: #332912;
    --profit: #34D399; --profit-soft: #0F2E22;
    --loss: #F87171; --loss-soft: #341616;
    --neutral: #8FA39B; --neutral-soft: #1B2A25;
    --shadow: 0 1px 2px rgba(0,0,0,0.3), 0 8px 24px -8px rgba(0,0,0,0.5);
  }
  * { box-sizing: border-box; }
  html { scroll-behavior: smooth; }
  body {
    margin: 0; background: var(--bg); color: var(--ink);
    font-family: "Manrope", system-ui, sans-serif; -webkit-font-smoothing: antialiased;
  }
  .tabular { font-variant-numeric: tabular-nums; }
  a { color: inherit; }
  section[id], .wrap { scroll-margin-top: 14px; }
  .wrap { max-width: 620px; margin: 0 auto; padding: 16px 16px 90px; }

  /* Header */
  .app-header { display: flex; align-items: center; justify-content: space-between; padding: 6px 4px 16px; }
  .app-header .brand { display: flex; align-items: center; gap: 10px; }
  .app-header .mark {
    width: 40px; height: 40px; border-radius: 12px; background: linear-gradient(135deg, var(--brand), #0D8F68);
    display: flex; align-items: center; justify-content: center; flex-shrink: 0;
  }
  .app-header .mark svg { width: 22px; height: 22px; }
  .app-header .name { font-size: 16.5px; font-weight: 800; margin: 0; }
  .app-header .name .accent { color: var(--brand); }
  .app-header .tagline { font-size: 11px; color: var(--ink-faint); margin: 1px 0 0; }
  .app-header .actions { display: flex; align-items: center; gap: 10px; }
  .bell { width: 36px; height: 36px; border-radius: 50%; background: var(--surface); border: 1px solid var(--border);
    display: flex; align-items: center; justify-content: center; font-size: 15px; position: relative; }
  .bell .dot { position: absolute; top: 6px; right: 7px; width: 7px; height: 7px; border-radius: 50%; background: var(--loss); }
  .avatar-chip { width: 36px; height: 36px; border-radius: 50%; background: var(--surface-2); border: 1px solid var(--border);
    display: flex; align-items: center; justify-content: center; font-size: 12.5px; font-weight: 800; color: var(--brand); }

  /* Status hero */
  .status-hero {
    background: linear-gradient(160deg, var(--brand-soft) 0%, var(--surface) 65%);
    border: 1px solid var(--border); border-radius: 20px; padding: 18px; box-shadow: var(--shadow); margin-bottom: 14px;
    display: flex; gap: 14px; align-items: flex-start;
  }
  .bot-avatar { width: 64px; height: 64px; border-radius: 18px; background: #0D1F1A; border: 1px solid var(--border);
    display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
  .bot-avatar svg { width: 42px; height: 42px; }
  .status-hero .mid { flex: 1; min-width: 0; }
  .running-pill { display: inline-flex; align-items: center; gap: 6px; background: rgba(52,211,153,0.14); color: var(--profit);
    font-size: 11px; font-weight: 800; padding: 4px 10px; border-radius: 999px; letter-spacing: 0.03em; }
  .running-pill .dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; animation: livepulse 2s infinite; }
  @keyframes livepulse { 0% { box-shadow: 0 0 0 0 rgba(52,211,153,0.5); } 70% { box-shadow: 0 0 0 6px rgba(52,211,153,0); } 100% { box-shadow: 0 0 0 0 rgba(52,211,153,0); } }
  .status-hero .headline { font-size: 15px; font-weight: 700; margin: 8px 0 4px; }
  .status-hero .subline { font-size: 12px; color: var(--ink-soft); display: flex; align-items: center; gap: 6px; }
  .practice-badge { background: var(--gold-soft); border: 1px solid rgba(227,179,65,0.3); border-radius: 14px; padding: 10px 12px;
    text-align: center; flex-shrink: 0; width: 116px; }
  .practice-badge .pb-title { font-size: 10px; font-weight: 800; color: var(--gold); letter-spacing: 0.04em; }
  .practice-badge .pb-body { font-size: 9.5px; color: var(--ink-soft); margin: 4px 0 6px; line-height: 1.4; }
  .practice-badge a { font-size: 10.5px; font-weight: 700; color: var(--brand); text-decoration: none; }

  /* Portfolio card */
  .portfolio-card { background: var(--surface); border: 1px solid var(--border); border-radius: 20px; padding: 20px;
    box-shadow: var(--shadow); margin-bottom: 14px; }
  .pf-label { font-size: 12.5px; color: var(--ink-soft); font-weight: 600; margin: 0; display: flex; align-items: center; gap: 5px; }
  .pf-balance { font-size: 34px; font-weight: 800; margin: 4px 0 0; letter-spacing: -0.01em; }
  .pf-change-row { display: flex; align-items: center; gap: 8px; margin: 6px 0 14px; flex-wrap: wrap; }
  .pf-change-pill { font-size: 12.5px; font-weight: 700; padding: 4px 10px; border-radius: 999px; }
  .pf-change-pill.profit { background: var(--profit-soft); color: var(--profit); }
  .pf-change-pill.loss { background: var(--loss-soft); color: var(--loss); }
  .pf-change-pill.neutral { background: var(--neutral-soft); color: var(--neutral); }
  .pf-started { font-size: 11.5px; color: var(--ink-faint); }
  .pf-body { display: flex; gap: 16px; align-items: stretch; }
  .pf-chart-wrap { flex: 1.5; min-width: 0; position: relative; }
  .pf-chart-badge { position: absolute; top: -6px; right: 4px; background: var(--surface-2); border: 1px solid var(--border);
    border-radius: 8px; padding: 3px 8px; font-size: 11px; font-weight: 700; }
  .pf-chart-labels { display: flex; justify-content: space-between; font-size: 10.5px; color: var(--ink-faint); margin-top: 2px; }
  .pf-stats { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 10px; justify-content: center; }
  .pf-stat { display: flex; align-items: center; gap: 8px; }
  .pf-stat-icon { width: 30px; height: 30px; border-radius: 9px; background: var(--surface-2); display: flex; align-items: center;
    justify-content: center; font-size: 13px; flex-shrink: 0; }
  .pf-stat-label { font-size: 10.5px; color: var(--ink-faint); margin: 0; }
  .pf-stat-value { font-size: 13.5px; font-weight: 700; margin: 1px 0 0; }

  /* Trade status card */
  .trade-status-card { background: var(--surface); border: 1px solid var(--border); border-radius: 18px; padding: 16px 18px;
    box-shadow: var(--shadow); margin-bottom: 18px; display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
  .ts-icon { width: 44px; height: 44px; border-radius: 12px; background: var(--brand-soft); color: var(--brand);
    display: flex; align-items: center; justify-content: center; font-size: 19px; flex-shrink: 0; }
  .ts-mid { flex: 1; min-width: 180px; }
  .ts-title { font-size: 14.5px; font-weight: 700; margin: 0 0 3px; }
  .ts-body { font-size: 12px; color: var(--ink-soft); margin: 0; line-height: 1.5; }
  .ts-next { text-align: right; flex-shrink: 0; }
  .ts-next-label { font-size: 10px; color: var(--ink-faint); font-weight: 700; text-transform: uppercase; letter-spacing: 0.03em; }
  .ts-next-value { font-size: 12.5px; font-weight: 700; margin-top: 2px; }

  /* Section header */
  .section-head { display: flex; align-items: center; justify-content: space-between; margin: 0 0 12px; }
  .section-head h2 { font-size: 15.5px; font-weight: 800; margin: 0; display: flex; align-items: center; gap: 8px; }
  .section-head .count-badge { background: var(--surface-2); border: 1px solid var(--border); border-radius: 999px;
    padding: 3px 10px; font-size: 11px; font-weight: 700; color: var(--ink-soft); }
  .section-head .view-all { font-size: 12px; font-weight: 700; color: var(--brand); text-decoration: none; }
  .section-head-right { display: flex; align-items: center; gap: 10px; }
  section.block { margin-bottom: 22px; }

  /* Empty/sample */
  .empty-card { background: var(--surface-2); border: 1px dashed var(--border); border-radius: 14px; padding: 18px;
    text-align: center; color: var(--ink-soft); font-size: 12.5px; line-height: 1.6; margin-bottom: 12px; }
  .empty-card strong { color: var(--ink); }
  .sample-banner { display: flex; align-items: center; gap: 8px; margin-bottom: 10px; }
  .sample-tag { font-size: 10px; font-weight: 800; letter-spacing: 0.05em; text-transform: uppercase; color: var(--gold);
    background: var(--gold-soft); padding: 4px 9px; border-radius: 7px; flex-shrink: 0; }
  .sample-note { font-size: 11px; color: var(--ink-faint); }
  .sample-block { opacity: 0.92; }

  /* Holdings */
  .holding-card { background: var(--surface); border: 1px solid var(--border); border-radius: 18px; padding: 16px;
    box-shadow: var(--shadow); margin-bottom: 12px; }
  .holding-top { display: flex; align-items: center; gap: 10px; margin-bottom: 14px; }
  .stock-avatar { width: 42px; height: 42px; border-radius: 12px; background: var(--brand-soft); color: var(--brand);
    display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 800; flex-shrink: 0; }
  .holding-name { flex: 1; min-width: 0; }
  .h-symbol { font-size: 14.5px; font-weight: 800; margin: 0; }
  .h-company { font-size: 11px; color: var(--ink-faint); margin: 1px 0 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .holding-tag { background: var(--profit-soft); color: var(--profit); font-size: 10.5px; font-weight: 700; padding: 4px 10px;
    border-radius: 999px; flex-shrink: 0; }
  .holding-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; padding: 12px; background: var(--surface-2);
    border-radius: 12px; margin-bottom: 12px; }
  .hg-label { font-size: 10px; color: var(--ink-faint); margin: 0 0 3px; }
  .hg-value { font-size: 13px; font-weight: 700; margin: 0; }
  .hg-value.profit { color: var(--profit); }
  .hg-value.loss { color: var(--loss); }
  .why-holding { display: flex; gap: 10px; background: var(--brand-soft); border-radius: 12px; padding: 11px 13px; margin-bottom: 12px; }
  .why-icon { font-size: 16px; flex-shrink: 0; }
  .why-title { font-size: 12px; font-weight: 700; color: var(--brand); margin: 0 0 2px; }
  .why-body { font-size: 11.5px; color: var(--ink-soft); margin: 0; line-height: 1.5; }
  .exit-plan { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
  .exit-card { display: flex; gap: 9px; padding: 11px; border-radius: 12px; border: 1px solid var(--border); }
  .exit-card.stop { background: rgba(248,113,113,0.06); }
  .exit-card.target { background: rgba(52,211,153,0.06); }
  .exit-icon { font-size: 15px; flex-shrink: 0; }
  .exit-label { font-size: 10.5px; color: var(--ink-faint); margin: 0; }
  .exit-value { font-size: 13.5px; font-weight: 800; margin: 2px 0 4px; }
  .exit-card.stop .exit-value { color: var(--loss); }
  .exit-card.target .exit-value { color: var(--profit); }
  .exit-caption { font-size: 10px; color: var(--ink-faint); margin: 0; line-height: 1.4; }

  /* Activity */
  .activity-row { display: flex; align-items: center; gap: 10px; padding: 12px; background: var(--surface); border: 1px solid var(--border);
    border-radius: 14px; margin-bottom: 9px; }
  .date-badge { width: 40px; text-align: center; flex-shrink: 0; }
  .date-badge .dn { display: block; font-size: 15px; font-weight: 800; }
  .date-badge .dm { display: block; font-size: 9px; color: var(--ink-faint); font-weight: 700; letter-spacing: 0.03em; }
  .activity-badge { width: 30px; height: 30px; border-radius: 50%; flex-shrink: 0; display: flex; align-items: center;
    justify-content: center; font-size: 11px; font-weight: 800; }
  .activity-badge.buy { background: var(--brand-soft); color: var(--brand); }
  .activity-badge.sell { background: var(--gold-soft); color: var(--gold); }
  .activity-main { flex: 1; min-width: 0; }
  .a-title { font-size: 13px; font-weight: 700; margin: 0; }
  .a-sub { font-size: 11px; color: var(--ink-faint); margin: 1px 0 0; }
  .activity-right { text-align: right; flex-shrink: 0; }
  .a-qty { font-size: 11.5px; font-weight: 700; margin: 0; }
  .a-price { font-size: 10.5px; color: var(--ink-faint); margin: 1px 0 4px; }
  .status-pill { font-size: 10.5px; font-weight: 700; padding: 3px 9px; border-radius: 999px; display: inline-block; }
  .status-pill.open { background: var(--neutral-soft); color: var(--neutral); }
  .status-pill.profit { background: var(--profit-soft); color: var(--profit); }
  .status-pill.loss { background: var(--loss-soft); color: var(--loss); }

  /* Results */
  .results-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
  .result-card { background: var(--surface); border: 1px solid var(--border); border-top: 3px solid var(--border);
    border-radius: 14px; padding: 13px; box-shadow: var(--shadow); }
  .result-card.profit { border-top-color: var(--profit); }
  .result-card.loss { border-top-color: var(--loss); }
  .result-card.neutral { border-top-color: var(--neutral); }
  .result-icon { font-size: 14px; margin-bottom: 6px; }
  .result-label { font-size: 10.5px; color: var(--ink-faint); font-weight: 700; margin: 0 0 5px; }
  .result-value { font-size: 16.5px; font-weight: 800; margin: 0; }
  .result-value.profit { color: var(--profit); }
  .result-value.loss { color: var(--loss); }
  .result-value.neutral { color: var(--neutral); }
  .result-sub { font-size: 10px; color: var(--ink-faint); margin: 4px 0 0; }

  /* Learn banner */
  .learn-banner { background: linear-gradient(120deg, var(--brand-soft), var(--surface)); border: 1px solid var(--border);
    border-radius: 18px; padding: 16px 18px; display: flex; align-items: center; gap: 14px; flex-wrap: wrap; margin-bottom: 10px; }
  .learn-icon { font-size: 22px; flex-shrink: 0; }
  .learn-mid { flex: 1; min-width: 0; }
  .learn-title { font-size: 13.5px; font-weight: 800; margin: 0 0 3px; }
  .learn-title .accent { color: var(--brand); }
  .learn-body { font-size: 11.5px; color: var(--ink-soft); margin: 0; }
  .learn-link { font-size: 12px; font-weight: 700; color: var(--brand); text-decoration: none; white-space: nowrap; flex-shrink: 0; }

  /* Trust points */
  .trust-points { display: flex; flex-direction: column; gap: 8px; margin-bottom: 20px; }
  .trust-points .row { display: flex; align-items: flex-start; gap: 8px; font-size: 11.5px; color: var(--ink-soft); line-height: 1.5; }
  .trust-points .check { color: var(--brand); flex-shrink: 0; font-weight: 800; }
  .trust-points strong { color: var(--ink); }

  /* Bottom tab bar */
  .tabbar { position: fixed; bottom: 0; left: 0; right: 0; background: var(--surface); border-top: 1px solid var(--border);
    display: flex; justify-content: space-around; padding: 8px 4px calc(8px + env(safe-area-inset-bottom)); z-index: 10; }
  .tabbar a { display: flex; flex-direction: column; align-items: center; gap: 3px; text-decoration: none; color: var(--ink-faint);
    font-size: 10px; font-weight: 700; padding: 4px 8px; }
  .tabbar a.active { color: var(--brand); }
  .tabbar a .tab-icon { font-size: 17px; }

  @media (min-width: 720px) {
    .wrap { max-width: 760px; padding-top: 28px; }
    .holding-grid { grid-template-columns: repeat(4, 1fr); }
  }
  @media (max-width: 480px) {
    .pf-body { flex-direction: column; }
    .exit-plan { grid-template-columns: 1fr; }
    .holding-grid { grid-template-columns: repeat(2, 1fr); row-gap: 14px; }
  }
</style>
</head>
<body>
<div class="wrap" id="top">
  <header class="app-header">
    <div class="brand">
      <div class="mark"><svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M4 19V13M10 19V9M16 19V5M22 19H2" stroke="white" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg></div>
      <div>
        <p class="name">Santha's <span class="accent">Trading Bot</span></p>
        <p class="tagline">Automated &bull; Smart &bull; Stress Free</p>
      </div>
    </div>
    <div class="actions">
      <div class="bell">&#128276;<span class="dot"></span></div>
      <div class="avatar-chip">SK</div>
    </div>
  </header>

  <section class="status-hero" id="bot-status">
    <div class="bot-avatar">
      <svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
        <ellipse cx="32" cy="58" rx="14" ry="3" fill="#000" opacity="0.25"/>
        <path d="M46 34 L54 27" stroke="#34D399" stroke-width="3" stroke-linecap="round"/>
        <circle cx="55" cy="25" r="3.2" fill="#F1F5F3"/>
        <rect x="12" y="20" width="40" height="32" rx="14" fill="#EFFBF5" stroke="#34D399" stroke-width="2"/>
        <circle cx="24" cy="35" r="4.2" fill="#0F2E22"/>
        <circle cx="40" cy="35" r="4.2" fill="#0F2E22"/>
        <path d="M24 44 Q32 50 40 44" stroke="#0F2E22" stroke-width="2.4" stroke-linecap="round" fill="none"/>
        <line x1="32" y1="20" x2="32" y2="11" stroke="#34D399" stroke-width="2.2" stroke-linecap="round"/>
        <circle cx="32" cy="7" r="3" fill="#34D399"/>
      </svg>
    </div>
    <div class="mid">
      <span class="running-pill"><span class="dot"></span>BOT IS RUNNING</span>
      <p class="headline">Watching the market automatically</p>
      <p class="subline">&#128202; Checking @@UNIVERSE_SIZE@@ NSE stocks for opportunities</p>
    </div>
    <div class="practice-badge">
      <p class="pb-title">&#129514; PRACTICE MODE</p>
      <p class="pb-body">No real money is being used.</p>
      <a href="#trust">Learn more &rarr;</a>
    </div>
  </section>

  <section class="portfolio-card">
    <p class="pf-label">Your Portfolio Value <span title="Cash plus the current value of everything you own">&#8505;&#65039;</span></p>
    <p class="pf-balance tabular">@@BALANCE@@</p>
    <div class="pf-change-row">
      <span class="pf-change-pill @@CHANGE_CLASS@@ tabular">@@CHANGE_ARROW@@ @@CHANGE_AMOUNT@@ (@@CHANGE_PCT@@)</span>
      <span class="pf-started">Started with @@STARTING_CAPITAL@@</span>
    </div>
    <div class="pf-body">
      <div class="pf-chart-wrap">
        <span class="pf-chart-badge tabular">@@CHART_BADGE@@</span>
        @@CHART_SVG@@
        <div class="pf-chart-labels"><span>Start</span><span>Now</span></div>
      </div>
      <div class="pf-stats">
        <div class="pf-stat"><div class="pf-stat-icon">&#128231;</div><div><p class="pf-stat-label">Available Cash</p><p class="pf-stat-value tabular">@@AVAILABLE_CASH@@</p></div></div>
        <div class="pf-stat"><div class="pf-stat-icon">&#128338;</div><div><p class="pf-stat-label">Invested Amount</p><p class="pf-stat-value tabular">@@INVESTED_AMOUNT@@</p></div></div>
        <div class="pf-stat"><div class="pf-stat-icon">&#128200;</div><div><p class="pf-stat-label">Total Profit / Loss</p><p class="pf-stat-value @@TOTAL_PL_CLASS@@ tabular">@@TOTAL_PL@@</p></div></div>
      </div>
    </div>
  </section>

  <section class="trade-status-card">
    <div class="ts-icon">&#128269;</div>
    <div class="ts-mid">
      <p class="ts-title">@@TRADE_STATUS_TITLE@@</p>
      <p class="ts-body">@@TRADE_STATUS_BODY@@</p>
    </div>
    <div class="ts-next">
      <p class="ts-next-label">&#128197; Next Check</p>
      <p class="ts-next-value">@@NEXT_CHECK@@</p>
    </div>
  </section>

  <section class="block" id="stocks">
    <div class="section-head">
      <h2>&#128230; Your Stocks</h2>
      <div class="section-head-right">
        <span class="count-badge">@@STOCK_COUNT@@ @@STOCK_WORD@@</span>
        <a class="view-all" href="#stocks">View all &rarr;</a>
      </div>
    </div>
    @@HOLDINGS_SECTION@@
  </section>

  <section class="block" id="activity">
    <div class="section-head">
      <h2>&#128337; What Your Bot Did</h2>
      <a class="view-all" href="#activity">View all activity &rarr;</a>
    </div>
    @@ACTIVITY_SECTION@@
  </section>

  <section class="block" id="results">
    <div class="section-head"><h2>&#128202; Your Results</h2></div>
    <div class="results-row">@@RESULTS_SECTION@@</div>
  </section>

  <section class="learn-banner">
    <span class="learn-icon">&#128161;</span>
    <div class="learn-mid">
      <p class="learn-title"><span class="accent">New to trading?</span> No problem!</p>
      <p class="learn-body">Your bot handles everything. You just watch the results.</p>
    </div>
    <a class="learn-link" href="#trust">Learn how it works &rarr;</a>
  </section>

  <div class="trust-points" id="trust">
    <div class="row"><span class="check">&#10003;</span><span><strong>Practice account.</strong> No real money is ever used &mdash; a safe way to see how your bot performs.</span></div>
    <div class="row"><span class="check">&#10003;</span><span><strong>Never forced.</strong> Your bot only trades when confident, and skips days when nothing looks safe.</span></div>
    <div class="row"><span class="check">&#10003;</span><span><strong>Not advice.</strong> Past results, practice or real, never guarantee future ones.</span></div>
  </div>
  <p style="text-align:center;font-size:10.5px;color:var(--ink-faint);margin:0 0 8px;">Updated @@AS_OF@@</p>
</div>

<nav class="tabbar">
  <a href="#top" class="active"><span class="tab-icon">&#127968;</span>Home</a>
  <a href="#results"><span class="tab-icon">&#128200;</span>Results</a>
  <a href="#stocks"><span class="tab-icon">&#128188;</span>Stocks</a>
  <a href="#bot-status"><span class="tab-icon">&#129302;</span>Bot</a>
  <a href="#trust"><span class="tab-icon">&#9881;&#65039;</span>Settings</a>
</nav>
</body>
</html>
"""


def write_customer_html(engine: PaperTradingEngine, config: Config, path: str) -> str:
    import os

    html = render_customer_html(engine, config)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path
