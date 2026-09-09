#!/usr/bin/env python3
"""
Command-line entry point for the AI-Assisted Trading Analysis & Paper
Trading system.

Subcommands:
  scan            Analyze a universe of symbols and print structured trade
                   reports (NO TRADE is the expected default outcome).
  paper-trade     Run one polling cycle of the paper trading engine: scan,
                   execute approved trades, manage open positions.
  backtest        Run the backtester on one symbol's historical data.
  walk-forward    Run walk-forward validation on one symbol's historical data.
  dashboard       Render the CLI dashboard, or write a static HTML snapshot.
  readiness       Show the real-money readiness checklist.
  account-check   Report affordability/ranking for a given account balance
                   (e.g. --capital 100) across the scanned universe -- spec
                   Part 10-13/25 small-account mode. `scan` also accepts an
                   optional --capital override for the same purpose.
  research        Cross-stock robustness report across a full universe
                   (spec Part 22), optionally with --survival for the
                   small-account survival simulation (spec Part 25).
  optimize        Parameter sensitivity diagnostic (spec Part 28) -- a
                   read-only stability report, never a parameter search
                   (nothing is applied to config; spec Part 29).

Live trading is never available from this CLI. See README.md.
"""

from __future__ import annotations

import argparse
import logging
import sys

from config.settings import load_config
from data.market_data import MarketDataProvider, DataUnavailableError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("main")


def _fetch_macro_daily(config, md: MarketDataProvider, period: str):
    """Spec Part 2: shared macro-context fetch, reused by every command that
    ultimately runs Backtester.run() (backtest/walk-forward/research/
    optimize). Fetched once per invocation (market-wide, not per-symbol).
    Gated exactly the way cmd_backtest always was: config.macro.enabled and
    config.providers.macro != "none". Returns None (not {}) when macro is
    off -- matching Backtester.run()'s own macro_daily=None default, so
    downstream behavior is byte-identical to macro being disabled. A fetch
    failure for one series is reported and that series alone is omitted --
    honest degradation, same principle as the index_daily fetch beside
    every call site of this helper."""
    if not (config.macro.enabled and (config.providers.macro or "yfinance").lower() != "none"):
        return None
    macro_daily = {}
    macro_tickers = {
        "us_vix": config.macro.us_vix_symbol, "india_vix": config.macro.india_vix_symbol,
        "crude": config.macro.crude_symbol, "usdinr": config.macro.usdinr_symbol,
    }
    for key, ticker in macro_tickers.items():
        try:
            macro_daily[key] = md.get_daily(ticker, period=period)
        except DataUnavailableError as exc:
            print(f"Warning: macro series {key} ({ticker}) unavailable: {exc}.")
    return macro_daily


def _make_ml_provider(config, args):
    """Spec Part 7: ML in live scanning is opt-in via --ml (mirroring
    `walk-forward`'s existing --ml flag) -- returns None (off, today's
    exact behavior) unless requested. See paper_trading/live_ml_provider.py."""
    if not getattr(args, "ml", False):
        return None
    from paper_trading.live_ml_provider import LiveTradeOutcomeMLProvider

    return LiveTradeOutcomeMLProvider(
        min_history_bars=config.ml.min_history_bars, min_auc=config.ml.min_auc,
        stop_atr_multiple=config.ml.stop_atr_multiple, target_atr_multiple=config.ml.target_atr_multiple,
        max_holding_days=config.ml.max_holding_days, n_splits=config.ml.n_splits,
    )


def cmd_scan(args) -> None:
    from paper_trading.engine import PaperTradingEngine
    from paper_trading.scanner import UniverseScanner

    config = load_config(args.config)
    if args.capital is not None:
        # Overrides the account balance used for sizing/affordability in
        # THIS run's reports (spec Part 10-13) -- never mutates the
        # underlying config file.
        config.paper_trading.starting_capital = args.capital
    engine = PaperTradingEngine(
        config, fetch_news=not args.no_news, fetch_social=not args.no_social,
        ml_provider=_make_ml_provider(config, args),
    )
    # UniverseScanner defaults to config.universe.symbols (the configured NIFTY
    # 50/Next 50/custom universe) when `symbols` is empty -- never a hardcoded
    # list -- and applies config.universe.scan_batch_size/scan_batch_delay_seconds/
    # scan_cache_ttl_seconds (spec section 19).
    scanner = UniverseScanner(engine)
    cycle = scanner.scan(symbols=args.symbols or None)
    if not cycle.results:
        print(f"No results ({cycle.symbols_skipped}/{cycle.symbols_requested} symbol(s) skipped -- "
              "data unavailable or an analysis error for all requested symbols).")
        return
    for r in cycle.results:
        print(r.report.render_text())
        print("=" * 80)
    print(f"Scanned {cycle.symbols_scanned}/{cycle.symbols_requested} symbol(s) "
          f"({cycle.symbols_skipped} skipped).")


def cmd_paper_trade(args) -> None:
    from paper_trading.engine import PaperTradingEngine
    from paper_trading.scanner import UniverseScanner

    config = load_config(args.config)
    engine = PaperTradingEngine(
        config, fetch_news=not args.no_news, fetch_social=not args.no_social,
        ml_provider=_make_ml_provider(config, args),
    )
    scanner = UniverseScanner(engine)

    print(f"Starting paper trading cycle. Mode: {'LIVE' if config.system.live_trading_enabled else 'PAPER'}")
    if config.system.live_trading_enabled:
        print("WARNING: live_trading_enabled=True in config, but this CLI's broker is PaperBroker only -- "
              "no real orders can or will be sent.")

    engine.manage_open_positions()
    engine.record_equity_snapshot()   # one snapshot per cycle, independent of how often account_summary() is polled
    cycle = scanner.scan(symbols=args.symbols or None, force_refresh=True)   # a trading cycle always wants fresh data
    executed = []
    for r in cycle.results:
        trade_id = engine.execute_if_approved(r)
        if trade_id:
            executed.append((r.symbol, trade_id))
        print(r.report.render_text())
        print("-" * 80)

    print(f"\nScanned {cycle.symbols_scanned}/{cycle.symbols_requested} symbol(s) ({cycle.symbols_skipped} skipped).")
    print(f"Executed {len(executed)} paper trade(s): {executed}")
    print("\nAccount summary:", engine.account_summary())


def cmd_backtest(args) -> None:
    from backtesting.backtester import Backtester, BacktestConfig

    config = load_config(args.config)
    md = MarketDataProvider()
    try:
        daily = md.get_daily(args.symbol, period=args.period)
    except DataUnavailableError as exc:
        print(f"Could not fetch data for {args.symbol}: {exc}")
        sys.exit(1)

    index_daily = None
    if config.universe.index_symbol:
        try:
            index_daily = md.get_daily(config.universe.index_symbol, period=args.period)
        except DataUnavailableError as exc:
            print(f"Warning: could not fetch benchmark {config.universe.index_symbol}: {exc}. "
                  f"Market-condition scoring will be UNKNOWN for this run.")
    else:
        print(f"Warning: no universe.index_symbol configured for market={config.system.market!r}. "
              f"Set it explicitly in config.yaml (e.g. '^NSEI' for NSE) -- market-condition "
              f"scoring will be UNKNOWN for this run.")

    # Spec Part 2: global + India macro context. Fetched once (not opt-in,
    # unlike --ml -- see config.macro.enabled), aligned per-bar inside
    # Backtester.run() the same way index_daily already is.
    macro_daily = _fetch_macro_daily(config, md, args.period)

    bt_config = BacktestConfig(
        brokerage_pct=config.backtesting.brokerage_pct,
        taxes_pct=config.backtesting.taxes_pct,
        slippage_pct=config.backtesting.slippage_pct,
        bid_ask_spread_pct=config.backtesting.bid_ask_spread_pct,
        initial_capital=config.backtesting.initial_capital,
    )
    backtester = Backtester(config=config, backtest_config=bt_config)
    result = backtester.run(
        args.symbol, daily, index_daily=index_daily, index_symbol=config.universe.index_symbol,
        macro_daily=macro_daily,
    )

    print(f"Backtest results for {args.symbol} ({args.period}):")
    print(f"  Data quality: {result.data_quality.render_text() if result.data_quality else 'N/A'}")
    if not result.is_valid:
        print("  RUN INVALID: data quality failed the run-level gate -- nothing below was simulated.")
        return
    print("  NOTE: runs the SAME multi-factor StrategyPipeline used by paper trading "
          "(technical + market condition + risk/volatility always; fundamentals/news/social "
          "only where a real historical provider was supplied -- none was here, so those "
          "components are excluded/unavailable rather than faked; see "
          "backtesting/historical_providers.py).")
    print(f"  Signals approved by the pipeline: {result.signals_approved} "
          f"(of which {result.signals_rejected_at_execution} failed the execution-time re-check "
          f"and were correctly skipped, not forced through)")
    for k, v in result.metrics.as_dict().items():
        print(f"  {k}: {v}")
    print("\nBy regime:")
    for regime, metrics in result.metrics_by_regime.items():
        print(f"  {regime}: trades={metrics.num_trades} win_rate={metrics.win_rate_pct:.1%} "
              f"profit_factor={metrics.profit_factor:.2f} expectancy={metrics.expectancy:.2f}")
    print("\nBenchmark comparison:")
    bc = result.benchmark_comparison
    if bc and bc.benchmark_available:
        print(f"  {bc.symbol or '(benchmark)'} buy-and-hold: {bc.buy_hold_return_pct:.2%} total "
              f"({bc.buy_hold_cagr_pct:.2%} CAGR) vs. strategy: {bc.strategy_return_pct:.2%} total "
              f"({bc.strategy_cagr_pct:.2%} CAGR) -- outperformance: {bc.outperformance_pct:.2%}")
        print(f"  Note: {bc.note}")
    else:
        print(f"  Not available: {bc.note if bc else 'no benchmark comparison computed'}")
    print("\nExecution assumptions (spec sections 11 & 24):")
    for name, description in result.assumptions.items():
        print(f"  - {name}: {description}")


def cmd_walk_forward(args) -> None:
    from backtesting.backtester import BacktestConfig
    from backtesting.walk_forward import WalkForwardValidator

    config = load_config(args.config)
    md = MarketDataProvider()
    try:
        daily = md.get_daily(args.symbol, period=args.period)
    except DataUnavailableError as exc:
        print(f"Could not fetch data for {args.symbol}: {exc}")
        sys.exit(1)

    index_daily = None
    if config.universe.index_symbol:
        try:
            index_daily = md.get_daily(config.universe.index_symbol, period=args.period)
        except DataUnavailableError:
            pass

    # Spec Part 2: same shared macro fetch as cmd_backtest -- previously
    # missing here, so every walk-forward fold silently ran with
    # macro_context=None even though Backtester.run() already supports it.
    macro_daily = _fetch_macro_daily(config, md, args.period)

    validator = WalkForwardValidator(
        config=config,
        backtest_config=BacktestConfig(initial_capital=config.backtesting.initial_capital),
        ml_enabled=args.ml,
    )
    report = validator.run(
        args.symbol, daily, index_daily=index_daily, macro_daily=macro_daily,
        train_bars=args.train_bars, validation_bars=args.validation_bars, test_bars=args.test_bars,
    )
    print(report.summary())
    print("NOTE: runs the SAME multi-factor StrategyPipeline used by paper trading "
          "(technical + market condition + risk/volatility always; fundamentals/news/social "
          "only where a real historical provider was supplied -- none was here, so those "
          "components are excluded/unavailable rather than faked; see "
          "backtesting/historical_providers.py).")

    from backtesting.health_report import assess_strategy_health
    health = assess_strategy_health(report)
    print()
    print(health.render_text())

    from backtesting.monte_carlo import run_monte_carlo
    oos_trades = [t for f in report.folds for t in f.out_of_sample_result.trades]
    if oos_trades:
        mc = run_monte_carlo(
            [t.net_pnl for t in oos_trades], initial_capital=config.backtesting.initial_capital,
            method="shuffle", seed=42,
        )
        print()
        print(mc.summary())
    else:
        print("\nMonte Carlo: skipped -- no out-of-sample trades were taken across any fold.")


def cmd_optimize(args) -> None:
    """
    Spec Part 28 (overfitting protection). This is a DIAGNOSTIC, never a
    search -- see backtesting/sensitivity.py's module docstring. Nothing
    here is applied to config.yaml or persisted anywhere.
    """
    from backtesting.sensitivity import DEFAULT_PARAMETER_SWEEPS, run_parameter_sensitivity

    config = load_config(args.config)
    md = MarketDataProvider()
    try:
        daily = md.get_daily(args.symbol, period=args.period)
    except DataUnavailableError as exc:
        print(f"Could not fetch data for {args.symbol}: {exc}")
        sys.exit(1)

    index_daily = None
    if config.universe.index_symbol:
        try:
            index_daily = md.get_daily(config.universe.index_symbol, period=args.period)
        except DataUnavailableError:
            pass

    # Spec Part 2: same shared macro fetch as cmd_backtest -- previously
    # missing here, so every sensitivity trial silently ran with
    # macro_context=None even though Backtester.run() already supports it.
    macro_daily = _fetch_macro_daily(config, md, args.period)

    sweeps = DEFAULT_PARAMETER_SWEEPS
    if args.parameters:
        requested = [p.strip() for p in args.parameters.split(",")]
        unknown = [p for p in requested if p not in DEFAULT_PARAMETER_SWEEPS]
        if unknown:
            print(f"Unknown parameter(s) {unknown}; choose from {list(DEFAULT_PARAMETER_SWEEPS)}.")
            sys.exit(1)
        sweeps = {p: DEFAULT_PARAMETER_SWEEPS[p] for p in requested}

    report = run_parameter_sensitivity(
        config, args.symbol, daily, index_daily=index_daily, macro_daily=macro_daily, sweeps=sweeps,
        train_bars=args.train_bars, validation_bars=args.validation_bars, test_bars=args.test_bars,
        period=args.period,
    )
    print(report.summary())
    print(
        "\nNothing above was applied anywhere -- this is a read-only diagnostic. If a parameter is "
        "flagged UNSTABLE, that is a reason for caution, not a signal to hand-pick the best-looking "
        "value from this table (spec Part 29: never select parameters using test results)."
    )
    print("NOTE: runs the SAME multi-factor StrategyPipeline used by paper trading "
          "(technical + market condition + risk/volatility always; fundamentals/news/social "
          "only where a real historical provider was supplied -- none was here, so those "
          "components are excluded/unavailable rather than faked; see "
          "backtesting/historical_providers.py).")


def cmd_dashboard(args) -> None:
    from paper_trading.engine import PaperTradingEngine
    from paper_trading.scanner import UniverseScanner
    from dashboard.cli_dashboard import render_cli_dashboard
    from dashboard.html_report import render_html_report

    config = load_config(args.config)
    engine = PaperTradingEngine(
        config, fetch_news=not args.no_news, fetch_social=not args.no_social,
        ml_provider=_make_ml_provider(config, args),
    )
    cycle = UniverseScanner(engine).scan(symbols=args.symbols or None)

    if args.html:
        path = render_html_report(engine, cycle.results, config, args.html, symbols_requested=cycle.symbols_requested)
        print(f"Wrote dashboard snapshot to {path}")
    else:
        render_cli_dashboard(engine, cycle.results, config, symbols_requested=cycle.symbols_requested)


def cmd_readiness(args) -> None:
    from risk.readiness import ReadinessChecklist

    checklist = ReadinessChecklist()
    print(checklist.render())
    print(
        "\nTo mark an item complete, edit your own operational checklist (this CLI intentionally "
        "does not expose a way to toggle these from the command line -- readiness should be "
        "attested deliberately, in writing, by whoever is accountable for the account)."
    )


def cmd_account_check(args) -> None:
    """
    Spec Part 10-13 & 25: for a given account balance, scan the configured
    (or explicitly given) universe and report which candidates are actually
    affordable/tradeable -- ranked by QUALITY + AFFORDABILITY (spec Part
    12), never by price alone. Reuses the exact same UniverseScanner ->
    PaperTradingEngine -> StrategyPipeline path as `scan`/`paper-trade` --
    no separate decision logic. Never places an order.
    """
    from paper_trading.candidate_ranking import affordable_candidates, rank_by_quality_and_affordability
    from paper_trading.engine import PaperTradingEngine
    from paper_trading.scanner import UniverseScanner

    config = load_config(args.config)
    config.paper_trading.starting_capital = args.capital
    engine = PaperTradingEngine(
        config, fetch_news=not args.no_news, fetch_social=not args.no_social,
        ml_provider=_make_ml_provider(config, args),
    )
    scanner = UniverseScanner(engine)
    cycle = scanner.scan(symbols=args.symbols or None)

    currency = config.paper_trading.base_currency
    print(f"Account-check for capital: {currency} {args.capital:,.2f}")
    print(f"Scanned {cycle.symbols_scanned}/{cycle.symbols_requested} symbol(s) "
          f"({cycle.symbols_skipped} skipped).\n")

    if not cycle.results:
        print("No results (data unavailable or an analysis error for every requested symbol).")
        return

    ranked = rank_by_quality_and_affordability(cycle.results, capital=args.capital)
    affordable = affordable_candidates(ranked)

    if not affordable:
        # Every candidate's raw price alone exceeds what this capital could
        # buy even 1 share of (a price-only estimate for symbols where no
        # direction was reached this cycle -- see candidate_ranking.py) --
        # genuinely a capital problem, not a "no good setup" problem.
        print("INSUFFICIENT CAPITAL FOR THIS MARKET/INSTRUMENT")
        print(
            "No candidate in the scanned universe could be purchased -- not even one whole "
            "share at its current price -- within this account's balance. This is the "
            "correct, honest outcome for a balance this small against these instruments' "
            "prices; it is not forced into a trade."
        )
        return

    priced_only = [c for c in affordable if c.affordability_basis == "price_only"]
    if not any(c.affordability_basis == "risk_assessment" for c in affordable) and not any(
        c.approved for c in ranked
    ):
        # Capital could afford SOME of these instruments by price, but no
        # symbol produced a qualifying directional signal this cycle at
        # all -- NO TRADE is this system's default, expected outcome most
        # of the time (see strategy/signal_engine.py), and must not be
        # reported as a capital problem.
        print(
            f"No qualifying trade signal this cycle ({len(priced_only)} candidate(s) were "
            "affordable by price alone -- see below). NO TRADE is the expected default outcome "
            "here, not a sign this account's capital is the limiting factor."
        )

    header = (
        f"{'SYMBOL':<15}{'PRICE':>10}{'AFFORD_QTY':>12}{'BASIS':>15}{'POS_SIZE':>10}{'DECISION':>10}"
        f"{'APPROVED':>10}{'CONF':>8}{'TECH%ILE':>10}{'ML_P':>8}{'EV':>10}{'R:R':>8}  REASON"
    )
    print(header)
    for c in ranked:
        ev_str = f"{c.expected_value_total:.2f}" if c.expected_value_total is not None else "N/A"
        rr_str = f"{c.risk_reward_ratio:.2f}" if c.risk_reward_ratio is not None else "N/A"
        pctile_str = f"{c.technical_percentile:.0f}" if c.technical_percentile is not None else "N/A"
        ml_str = f"{c.ml_probability:.0%}" if c.ml_probability is not None else "N/A"
        print(
            f"{c.symbol:<15}{c.price:>10.2f}{c.affordable_quantity:>12}{c.affordability_basis:>15}"
            f"{c.position_size:>10}{c.decision:>10}{str(c.approved):>10}{c.confidence:>8.1f}"
            f"{pctile_str:>10}{ml_str:>8}{ev_str:>10}{rr_str:>8}  {c.rejection_reason or ''}"
        )
    print(f"\n{len(affordable)}/{len(ranked)} candidate(s) affordable; "
          f"{sum(1 for c in ranked if c.approved)}/{len(ranked)} fully approved. "
          "('price_only' basis = no directional signal reached this cycle, so affordability "
          "is a raw price/capital estimate, not a full risk-costed one. TECH%ILE = this "
          "symbol's technical score percentile-ranked against the rest of THIS scan cycle only "
          "-- informational, never a substitute for the approval gate above.)")


def _resolve_research_symbols(args, config) -> list:
    """--symbols wins if given; else --universe selects a preset; else the
    configured universe (spec's "never a hardcoded list" principle applies
    here too -- 'configured' is the DEFAULT, not nifty50)."""
    if args.symbols:
        return list(args.symbols)
    if args.universe == "nifty50":
        from config.nse_universe import NIFTY_50
        return list(NIFTY_50)
    if args.universe == "nifty_next_50":
        from config.nse_universe import NIFTY_NEXT_50
        return list(NIFTY_NEXT_50)
    return list(config.universe.symbols)


def cmd_research(args) -> None:
    """
    Spec Part 22 (cross-stock robustness) & Part 25 (small-account
    survival, with --survival). Never optimizes on or special-cases any
    individual symbol -- loops the same Backtester every backtest command
    uses, across the whole requested universe.
    """
    from backtesting.universe_backtest import run_universe_backtest
    from config.nse_universe import NIFTY_50_SECTOR_MAP

    config = load_config(args.config)
    symbols = _resolve_research_symbols(args, config)
    print(f"Cross-stock robustness: {len(symbols)} symbol(s), period={args.period}, "
          f"universe={'--symbols' if args.symbols else args.universe}.\n")

    # A shared MarketDataProvider (rather than letting run_universe_backtest/
    # run_small_account_survival each build their own) so macro context
    # (spec Part 2, market-wide -- fetched ONCE, not per symbol) is fetched
    # a single time and reused across both passes, mirroring how index_daily
    # is already fetched once per pass inside those functions.
    md = MarketDataProvider()
    macro_daily = _fetch_macro_daily(config, md, args.period)

    report = run_universe_backtest(
        config, symbols, period=args.period, sector_map=NIFTY_50_SECTOR_MAP,
        market_data=md, macro_daily=macro_daily,
    )
    print(report.summary())

    if args.survival:
        from backtesting.small_account_survival import SMALL_ACCOUNT_CAPITAL_LEVELS, run_small_account_survival

        capital_levels = SMALL_ACCOUNT_CAPITAL_LEVELS
        if args.capital_levels:
            capital_levels = [float(x) for x in args.capital_levels.split(",")]

        print(f"\nRunning small-account survival across {len(capital_levels)} capital level(s) "
              f"(this re-runs backtests per level -- may take a while for a large universe)...\n")
        survival = run_small_account_survival(
            config, symbols, capital_levels=capital_levels, period=args.period,
            market_data=md, macro_daily=macro_daily,
        )
        print(survival.summary())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI-Assisted Trading Analysis & Paper Trading System")
    parser.add_argument("--config", default=None, help="Path to a config.yaml override file.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_scan = sub.add_parser("scan", help="Analyze a universe of symbols.")
    p_scan.add_argument("symbols", nargs="*", help="Symbols to scan (defaults to config universe).")
    p_scan.add_argument("--no-news", action="store_true", help="Skip news analysis (faster, offline-friendly).")
    p_scan.add_argument("--no-social", action="store_true", help="Skip social sentiment analysis.")
    p_scan.add_argument("--capital", type=float, default=None,
                         help="Override the account balance used for sizing/affordability "
                              "in this run's reports (spec Part 10-13).")
    p_scan.add_argument("--ml", action="store_true",
                         help="Opt in to the live ML trade-outcome model (spec Part 7); fit once per "
                              "symbol and cached for this run. Off by default.")
    p_scan.set_defaults(func=cmd_scan)

    p_paper = sub.add_parser("paper-trade", help="Run one paper-trading cycle.")
    p_paper.add_argument("symbols", nargs="*")
    p_paper.add_argument("--no-news", action="store_true")
    p_paper.add_argument("--no-social", action="store_true")
    p_paper.add_argument("--ml", action="store_true", help="Opt in to the live ML trade-outcome model (spec Part 7).")
    p_paper.set_defaults(func=cmd_paper_trade)

    p_bt = sub.add_parser("backtest", help="Backtest one symbol.")
    p_bt.add_argument("symbol")
    p_bt.add_argument("--period", default="5y")
    p_bt.set_defaults(func=cmd_backtest)

    p_wf = sub.add_parser("walk-forward", help="Walk-forward validate one symbol.")
    p_wf.add_argument("symbol")
    p_wf.add_argument("--period", default="5y")
    p_wf.add_argument("--train-bars", type=int, default=500)
    p_wf.add_argument("--validation-bars", type=int, default=100)
    p_wf.add_argument("--test-bars", type=int, default=100)
    p_wf.add_argument("--ml", action="store_true", help="Fit an ML baseline per fold on train+validation only, "
                                                          "then score out-of-sample (hard leakage guard).")
    p_wf.set_defaults(func=cmd_walk_forward)

    p_dash = sub.add_parser("dashboard", help="Render the dashboard (terminal or HTML snapshot).")
    p_dash.add_argument("symbols", nargs="*")
    p_dash.add_argument("--html", default=None, help="Write a static HTML snapshot to this path instead of the terminal UI.")
    p_dash.add_argument("--no-news", action="store_true")
    p_dash.add_argument("--no-social", action="store_true")
    p_dash.add_argument("--ml", action="store_true", help="Opt in to the live ML trade-outcome model (spec Part 7).")
    p_dash.set_defaults(func=cmd_dashboard)

    p_ready = sub.add_parser("readiness", help="Show the real-money readiness checklist.")
    p_ready.set_defaults(func=cmd_readiness)

    p_account = sub.add_parser(
        "account-check",
        help="Report affordability/ranking for a given account balance across the scanned universe.",
    )
    p_account.add_argument("symbols", nargs="*", help="Symbols to scan (defaults to config universe).")
    p_account.add_argument("--capital", type=float, required=True, help="Account balance to evaluate.")
    p_account.add_argument("--no-news", action="store_true")
    p_account.add_argument("--no-social", action="store_true")
    p_account.add_argument("--ml", action="store_true", help="Opt in to the live ML trade-outcome model (spec Part 7).")
    p_account.set_defaults(func=cmd_account_check)

    p_research = sub.add_parser(
        "research", help="Cross-stock robustness report across a full universe (spec Part 22).",
    )
    p_research.add_argument("--symbols", nargs="*", default=None,
                             help="Explicit symbol list -- overrides --universe.")
    p_research.add_argument("--universe", choices=["configured", "nifty50", "nifty_next_50"], default="configured",
                             help="Which preset universe to scan when --symbols isn't given "
                                  "(default: config.universe.symbols -- never a hardcoded list).")
    p_research.add_argument("--period", default="5y")
    p_research.add_argument("--survival", action="store_true",
                             help="Also run the small-account survival simulation across 8 capital levels "
                                  "(spec Part 25) -- re-runs backtests per level, so this is slower.")
    p_research.add_argument("--capital-levels", default=None,
                             help="Comma-separated override for --survival's capital levels, e.g. "
                                  "'100,1000,100000'. Defaults to spec Part 25's 8-level list.")
    p_research.set_defaults(func=cmd_research)

    p_optimize = sub.add_parser(
        "optimize",
        help="Parameter sensitivity diagnostic (spec Part 28) -- read-only, never applies a value.",
    )
    p_optimize.add_argument("symbol")
    p_optimize.add_argument("--period", default="5y")
    p_optimize.add_argument("--train-bars", type=int, default=500)
    p_optimize.add_argument("--validation-bars", type=int, default=100)
    p_optimize.add_argument("--test-bars", type=int, default=100)
    p_optimize.add_argument("--parameters", default=None,
                             help="Comma-separated subset of parameters to sweep (default: all of "
                                  "min_confidence_to_trade, min_risk_reward, max_model_disagreement).")
    p_optimize.set_defaults(func=cmd_optimize)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
