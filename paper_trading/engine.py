"""
Paper Trading Engine.

Wires together every module built so far into one pipeline that runs
against LIVE/current market data but never places a real order:

  MarketDataProvider -> TechnicalAnalyzer -> FundamentalAnalyzer ->
  NewsAnalyzer -> SocialSentimentAnalyzer -> SignalInputs
      -> StrategyPipeline.decide()  [SignalEngine -> TradeRiskCalculator
         -> CapitalProtection.pre_trade_check -> TradeFilter]
      -> (approved?) -> PaperBroker.place_order -> TradeJournal

`self.pipeline` (a `strategy.pipeline.StrategyPipeline`, built by the same
`strategy.pipeline.build_pipeline(config)` factory that
`backtesting.backtester.Backtester` uses) is the SAME object type/
construction path used for backtesting -- see `strategy/pipeline.py`'s
module docstring and `tests/test_pipeline_parity.py` for the proof that a
backtest and a paper-trading scan run identical decision logic, not just
similarly-configured copies of it.

This is intentionally the "reference wiring" for the whole system: main.py's
`scan` and `paper-trade` commands call into this class, and a live-trading
engine (once section 20's readiness checklist passes) would reuse the exact
same pipeline up to the broker call.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from broker.broker_interface import Position
from broker.paper_broker import PaperBroker
from config.settings import Config
from data.data_quality import DataQualityReport
from data.macro_data import MacroDataProvider
from data.market_data import MarketDataProvider, DataUnavailableError, classify_regime
from fundamentals.fundamental_analysis import FundamentalAnalyzer, FundamentalSnapshot
from indicators.technical import TechnicalAnalyzer, TechnicalSnapshot
from news.news_analysis import NewsAnalyzer
from paper_trading.decision_log import DecisionLog, DecisionLogEntry
from paper_trading.journal import JournalEntry, TradeJournal
from paper_trading.live_ml_provider import LiveTradeOutcomeMLProvider
from risk.risk_engine import CapitalProtection, RiskAssessment
from sentiment.social_sentiment import SocialSentimentAnalyzer
from strategy.pipeline import StrategyPipeline, build_pipeline
from strategy.report import TradeReport
from strategy.signal_engine import SignalDecision, SignalInputs
from strategy.trade_filter import FilterResult

logger = logging.getLogger(__name__)


@dataclass
class ScanResult:
    symbol: str
    report: TradeReport
    signal: SignalDecision
    filter_result: FilterResult
    data_quality: Optional[DataQualityReport] = None
    # The RiskAssessment computed for this decision (None when the signal
    # engine never reached risk sizing at all, e.g. a HOLD/NO-TRADE that
    # never proposed a direction, or the data-quality gate fired first).
    # Carries position_size/affordable_quantity/EV/R:R -- consumed by
    # paper_trading/candidate_ranking.py and the decision log.
    risk: Optional[RiskAssessment] = None
    # The sector this symbol was scanned with (from UniverseScanner's
    # sector_map, threaded into scan_symbol) -- carried through so
    # execute_if_approved() can register/release sector exposure
    # correctly, without requiring every caller to separately re-supply
    # a sector value it has no other way to recover.
    sector: Optional[str] = None


class PaperTradingEngine:
    def __init__(
        self,
        config: Config,
        market_data: Optional[MarketDataProvider] = None,
        journal: Optional[TradeJournal] = None,
        decision_log: Optional[DecisionLog] = None,
        fetch_news: bool = True,
        fetch_social: bool = True,
        technical_analyzer: Optional[TechnicalAnalyzer] = None,
        fundamental_analyzer: Optional[FundamentalAnalyzer] = None,
        news_analyzer: Optional[NewsAnalyzer] = None,
        social_analyzer: Optional[SocialSentimentAnalyzer] = None,
        pipeline: Optional[StrategyPipeline] = None,
        ml_provider: Optional[LiveTradeOutcomeMLProvider] = None,
        macro_provider: Optional[MacroDataProvider] = None,
    ):
        # Every analyzer can be injected (used heavily by the test suite to
        # avoid real network calls); defaults are the real, network-backed
        # implementations used in normal operation.
        from config.providers import (
            make_fundamental_analyzer, make_macro_data_provider, make_market_data_provider,
            make_news_analyzer, make_social_analyzer,
        )

        self.config = config
        # Each default is resolved via config/providers.py, i.e. it respects
        # config.providers.* (itself settable via MARKET_DATA_PROVIDER /
        # NEWS_PROVIDER / FUNDAMENTALS_PROVIDER / SOCIAL_PROVIDER /
        # MACRO_PROVIDER env vars -- spec section 21) -- explicit injection
        # here (used heavily by the test suite) always wins over that
        # resolution.
        self.market_data = market_data or make_market_data_provider(config)
        self.technical_analyzer = technical_analyzer or TechnicalAnalyzer()
        self.fundamental_analyzer = fundamental_analyzer or make_fundamental_analyzer(config)
        self.news_analyzer = news_analyzer or make_news_analyzer(config)
        self.social_analyzer = social_analyzer or make_social_analyzer(config)
        # Spec Part 2: unlike ML (opt-in, None by default -- fit cost),
        # macro context is core analysis, always attempted unless
        # config.macro.enabled=False or providers.macro="none" (both
        # already resolved by make_macro_data_provider into a real no-op).
        self.macro_provider = macro_provider or make_macro_data_provider(config, market_data=self.market_data)
        # THE key line for architectural parity with backtesting: same
        # factory, same Config -> identical SignalEngine/TradeRiskCalculator/
        # TradeFilter construction as Backtester uses. Pass `pipeline`
        # explicitly to share the literal same instance across a
        # PaperTradingEngine and a Backtester in a test.
        self.pipeline: StrategyPipeline = pipeline or build_pipeline(config)
        self.capital_protection = CapitalProtection(
            starting_capital=config.paper_trading.starting_capital,
            max_daily_loss_pct=config.risk.max_daily_loss_pct,
            max_weekly_loss_pct=config.risk.max_weekly_loss_pct,
            max_simultaneous_positions=config.risk.max_simultaneous_positions,
            max_sector_exposure_pct=config.risk.max_sector_exposure_pct,
            max_single_position_pct=config.risk.max_single_position_pct,
            max_consecutive_losses_before_reduction=config.risk.max_consecutive_losses_before_reduction,
            risk_reduction_factor=config.risk.risk_reduction_factor,
        )
        self.broker = PaperBroker(
            starting_capital=config.paper_trading.starting_capital,
            market_data=self.market_data,
            slippage_pct=config.risk.slippage_pct,
            fee_pct=config.risk.transaction_cost_pct,
        )
        self.journal = journal or TradeJournal()
        # Spec section 17: "record every signal ... record rejected trades ...
        # every NO TRADE decision should include a reason." TradeJournal above
        # only records trades that were actually OPENED; this is the wider
        # record of every scan decision, approved or not.
        self.decision_log = decision_log or DecisionLog()
        # Spec Part 7: off by default (None) -- identical to every prior
        # release's behavior. Set via `--ml` (main.py) to opt in; see
        # paper_trading/live_ml_provider.py for why this is cached, not
        # re-fit on every scan.
        self.ml_provider = ml_provider
        self.fetch_news = fetch_news
        self.fetch_social = fetch_social
        self._trade_ids: Dict[str, str] = {}  # symbol -> journal trade_id, for open positions
        # Equity snapshots for drawdown tracking (spec section 12/17/18) --
        # PaperBroker itself has no notion of time, only current state, so
        # the engine is what remembers equity over the course of a session.
        self._equity_history: List[Tuple[datetime, float]] = []
        # Every real run of this engine (main.py's `paper-trade`) is a
        # single one-shot cycle, not a long-lived process -- so cash/
        # positions/capital-protection state must be rebuilt from the
        # journal's own persisted history on every startup, or each new
        # run would silently look like a brand-new account. A no-op when
        # the journal is empty (fresh account, or any test using an
        # isolated tmp_path journal).
        self._restore_state_from_journal()

    def _restore_state_from_journal(self) -> None:
        """Replays the journal's history through the SAME `CapitalProtection.
        register_open`/`register_close` methods already used for real-time
        updates, and reconstructs `PaperBroker`'s cash/positions with the
        same entry-leg cash formulas `place_order` itself uses -- see
        Phase 20's plan for why this reproduces the correct state rather
        than inventing new math. Closed trades are replayed in
        chronological order specifically so `CapitalProtection`'s internal
        day/week rollover lands correctly (a trade closed last week must
        not count toward this week's weekly P&L)."""
        closed = sorted(self.journal.closed_trades(), key=lambda e: e.exit_time)
        for e in closed:
            if e.net_pnl is not None:
                self.broker._cash += e.net_pnl
            self.capital_protection.register_close(
                e.sector, e.entry_price * e.quantity, e.net_pnl or 0.0,
                today=e.exit_time.date(),
            )

        for e in self.journal.open_trades():
            notional = e.entry_price * e.quantity
            if e.side.upper() == "BUY":
                self.broker._cash -= (notional + e.fees)
                pos_side = "LONG"
            else:
                self.broker._cash += (notional - e.fees)
                pos_side = "SHORT"
            self.broker._positions[e.symbol] = Position(
                symbol=e.symbol, side=pos_side, quantity=e.quantity,
                average_price=e.entry_price, stop_loss=e.stop_loss, target=e.target,
                opened_at=e.entry_time, sector=e.sector,
            )
            self._trade_ids[e.symbol] = e.trade_id
            self.capital_protection.register_open(e.sector, notional)

    # ------------------------------------------------------------------
    def scan_symbol(self, symbol: str, sector: Optional[str] = None) -> Optional[ScanResult]:
        # get_daily_with_quality never raises -- a totally failed fetch comes
        # back as (None, DataQualityReport(status="INVALID", ...)) instead of
        # DataUnavailableError, so a provider outage surfaces as an explicit
        # status rather than an exception (spec section 2). A symbol with no
        # data at all still can't be scanned (there's nothing to build a
        # SignalInputs from), so it's skipped from this cycle's candidates,
        # same as before -- but data that DOES exist and is merely
        # LOW-QUALITY (stale, gappy, invalid bars) is handled below by
        # StrategyPipeline's data-quality gate, which forces NO TRADE rather
        # than skipping the symbol silently.
        daily, quality_report = self.market_data.get_daily_with_quality(symbol)
        if daily is None:
            logger.warning("Skipping %s: %s", symbol, quality_report.render_text())
            self._log_decision(
                symbol, sector, decision="NO TRADE", approved=False, confidence=0.0,
                confidence_label="NO TRADE", direction="NONE", data_quality_status=quality_report.status,
                reasons=[quality_report.render_text()], account_balance=self.broker.get_balance(),
            )
            return None

        try:
            technical = self.technical_analyzer.analyze(symbol, daily)
        except ValueError as exc:
            logger.warning("Skipping %s: %s", symbol, exc)
            self._log_decision(
                symbol, sector, decision="NO TRADE", approved=False, confidence=0.0,
                confidence_label="NO TRADE", direction="NONE", data_quality_status=quality_report.status,
                reasons=[f"Technical analysis could not run: {exc}"], account_balance=self.broker.get_balance(),
            )
            return None

        market_trend = "UNKNOWN"
        relative_strength = float("nan")
        if self.config.universe.index_symbol:
            try:
                index_daily = self.market_data.get_daily(self.config.universe.index_symbol)
                market_trend = self.market_data.market_trend(index_daily)
                relative_strength = self.market_data.relative_strength(daily, index_daily)
            except DataUnavailableError:
                logger.warning("Index data unavailable; market context will be UNKNOWN for %s.", symbol)
        else:
            logger.warning(
                "No universe.index_symbol configured (market=%r has no benchmark preset) -- "
                "market trend/relative strength will be reported as UNKNOWN for %s. Set "
                "universe.index_symbol explicitly in config.yaml.", self.config.system.market, symbol,
            )

        sector_trend = "N/A"
        sector_index = self.config.universe.sector_indices.get(sector) if sector else None
        if sector_index:
            try:
                sector_daily = self.market_data.get_daily(sector_index)
                sector_trend = self.market_data.market_trend(sector_daily)
            except DataUnavailableError:
                logger.warning("Sector index data unavailable for sector %r.", sector)

        try:
            fundamentals = self.fundamental_analyzer.analyze(symbol)
        except Exception as exc:
            # Belt-and-suspenders: FundamentalAnalyzer.analyze() already
            # degrades an internal fetch failure to "unavailable" rather
            # than raising, but a custom-injected fundamental_analyzer
            # might not -- a fundamentals failure must never crash the scan
            # or block it from reaching the data-quality/NO-TRADE gate
            # (spec section 22). Fall back to an explicitly "all missing"
            # snapshot rather than fabricating any value.
            logger.warning("Fundamental analysis failed for %s (treating as unavailable): %s", symbol, exc)
            fundamentals = FundamentalSnapshot(
                symbol=symbol, revenue_growth=None, earnings_growth=None, eps=None,
                pe_ratio=None, pb_ratio=None, debt_to_equity=None, roe=None,
                profit_margin=None, operating_cash_flow=None, free_cash_flow=None,
                last_update=None, data_quality=0.0, is_stale=True, roce=None,
            )

        news = None
        if self.fetch_news:
            try:
                news = self.news_analyzer.analyze(symbol)
            except Exception as exc:
                logger.warning("News analysis failed for %s: %s", symbol, exc)

        social = None
        if self.fetch_social:
            try:
                social = self.social_analyzer.analyze(symbol)
            except Exception as exc:
                logger.warning("Social sentiment analysis failed for %s: %s", symbol, exc)

        avg_volume = self.market_data.average_volume(daily)
        # Live-scan regime: classify AS OF the last available bar (spec
        # Part 5.6) -- the SAME classifier backtesting uses per-bar
        # point-in-time (data/market_data.py's classify_regime, relocated
        # from backtesting/backtester.py so both callers share it).
        market_regime = classify_regime(daily)

        ml_probability_up = None
        if self.ml_provider is not None:
            try:
                ml_probability_up = self.ml_provider.predict(symbol, daily)
            except Exception as exc:
                logger.warning("ML prediction failed for %s (treating as unavailable): %s", symbol, exc)

        # Spec Part 2: cheap after the first call in a scan cycle -- the
        # provider's own short-TTL cache means a 50-symbol scan fetches
        # these 4 market-wide series once, not per symbol.
        try:
            macro_context = self.macro_provider.get_snapshot()
        except Exception as exc:
            logger.warning("Macro snapshot failed (treating as unavailable): %s", exc)
            macro_context = None

        inputs = SignalInputs(
            symbol=symbol,
            technical=technical,
            market_trend=market_trend,
            relative_strength=relative_strength,
            fundamentals=fundamentals,
            news=news,
            social=social,
            avg_volume_20d=avg_volume,
            min_liquidity_avg_volume=self.config.risk.min_liquidity_avg_volume,
            atr_pct_of_price=technical.atr_pct_of_price,
            max_atr_pct_of_price=self.config.risk.max_atr_pct_of_price,
            min_atr_pct_of_price=self.config.risk.min_atr_pct_of_price,
            history_bars=len(daily),
            market_regime=market_regime,
            ml_probability_up=ml_probability_up,
            macro_context=macro_context,
        )

        current_price = self.market_data.get_quote(symbol)

        def account_check(risk_assessment):
            notional = risk_assessment.notional_exposure if risk_assessment else 0.0
            return self.capital_protection.pre_trade_check(symbol=symbol, sector=sector, notional_exposure=notional)

        # THE call shared with Backtester.run() -- see strategy/pipeline.py.
        # risk_multiplier (spec Part 26): shrinks sizing after a losing
        # streak; CapitalProtection owns that state for this account.
        result = self.pipeline.decide(
            inputs, current_price=current_price, capital=self.broker.get_balance(),
            account_check_fn=account_check, market_trend=market_trend, sector_trend=sector_trend,
            data_quality=quality_report, risk_multiplier=self.capital_protection.current_risk_multiplier(),
        )
        # Spec Part 4/3 wired into the decision log (Phase 9): only
        # surfaced when the SAME gate signal_engine.py used actually let
        # the component count -- "fundamentals"/"news_sentiment" being
        # present in component_scores IS that gate's own answer, so this
        # reuses it rather than re-deriving fundamentals_ok/news_ok here.
        fundamental_risk = None
        if "fundamentals" in result.signal.component_scores and inputs.fundamentals is not None:
            fundamental_risk = inputs.fundamentals.fundamental_risk_score()
        news_sentiment_class = None
        if "news_sentiment" in result.signal.component_scores and inputs.news is not None:
            news_sentiment_class = inputs.news.sentiment_class

        self._log_decision(
            symbol, sector, decision=result.filter_result.final_decision,
            approved=result.filter_result.approved, confidence=result.signal.overall_confidence,
            confidence_label=result.signal.confidence_label, direction=result.signal.direction,
            data_quality_status=quality_report.status,
            reasons=result.filter_result.reasons or result.signal.reasons,
            account_balance=self.broker.get_balance(), price=current_price, risk=result.risk,
            component_scores=result.signal.component_scores, ml_probability=inputs.ml_probability_up,
            market_regime=result.signal.market_regime,
            fundamental_risk=fundamental_risk, news_sentiment_class=news_sentiment_class,
            technical=technical,
        )
        return ScanResult(
            symbol=symbol, report=result.report, signal=result.signal,
            filter_result=result.filter_result, data_quality=quality_report, risk=result.risk,
            sector=sector,
        )

    def _log_decision(
        self, symbol: str, sector: Optional[str], decision: str, approved: bool, confidence: float,
        confidence_label: str, direction: str, data_quality_status: str, reasons: List[str],
        account_balance: Optional[float] = None, price: Optional[float] = None,
        risk: Optional["RiskAssessment"] = None, component_scores: Optional[Dict[str, float]] = None,
        ml_probability: Optional[float] = None, market_regime: str = "UNKNOWN",
        fundamental_risk: Optional[float] = None, news_sentiment_class: Optional[str] = None,
        technical: Optional[TechnicalSnapshot] = None,
    ) -> None:
        """Spec section 17 / Part 18: every scan decision -- approved,
        rejected, or NO TRADE for lack of data -- is recorded with its
        reason(s) and the full component/risk breakdown that produced it,
        never just returned in-memory and forgotten if nobody happened to
        read this call's return value.

        `sector_score` is always None here -- there is no separate sector-
        index SCORE anywhere in this system (only a `sector_trend` text
        label, which this log doesn't carry a numeric column for).
        `momentum_score` is `technical.momentum_score()` when a
        `TechnicalSnapshot` was passed (the real-decision call site below
        always has one in scope) -- `None` only for the two early-return
        call sites in `scan_symbol` where a `TechnicalSnapshot` was never
        built (the data-quality gate or `TechnicalAnalyzer.analyze()`
        itself failed first, so there's genuinely nothing to compute it
        from). `fundamental_risk`/`news_sentiment_class` (Phase 9) ARE
        wired through, from `inputs.fundamentals`/`inputs.news` at the
        call site.
        """
        component_scores = component_scores or {}
        rejection_reason = reasons[0] if reasons else None
        self.decision_log.record(DecisionLogEntry(
            timestamp=datetime.now(timezone.utc), symbol=symbol, decision=decision, approved=approved,
            confidence=confidence, confidence_label=confidence_label, direction=direction,
            data_quality_status=data_quality_status,
            reasons="; ".join(reasons) if reasons else "(no reasons recorded)",
            sector=sector,
            account_balance=account_balance,
            price=price,
            affordable_quantity=risk.affordable_quantity if risk else None,
            technical_score=component_scores.get("technical"),
            fundamental_score=component_scores.get("fundamentals"),
            market_score=component_scores.get("market_condition"),
            news_score=component_scores.get("news_sentiment"),
            sentiment_score=component_scores.get("social_sentiment"),
            volume_score=component_scores.get("volume_price_behavior"),
            risk_score=component_scores.get("risk_volatility"),
            sector_score=None,    # no sector-relative score exists anywhere in this system
            momentum_score=technical.momentum_score() if technical is not None else None,
            ml_probability=ml_probability,
            risk_reward_ratio=risk.risk_reward_ratio if risk else None,
            expected_value=risk.expected_value_total if risk else None,
            estimated_cost=(risk.expected_transaction_cost + risk.expected_slippage_cost) if risk else None,
            position_size=risk.position_size if risk else None,
            rejection_reason=rejection_reason,
            market_regime=market_regime,
            fundamental_risk=fundamental_risk,
            news_sentiment_class=news_sentiment_class,
        ))

    def scan_universe(self, symbols: List[str], sector_map: Optional[Dict[str, str]] = None) -> List[ScanResult]:
        sector_map = sector_map or self.config.universe.sector_map
        results = []
        for symbol in symbols:
            result = self.scan_symbol(symbol, sector=sector_map.get(symbol))
            if result is not None:
                results.append(result)
        return results

    # ------------------------------------------------------------------
    def execute_if_approved(self, result: ScanResult) -> Optional[str]:
        """Places a paper order ONLY if the trade filter approved it. Returns
        the journal trade_id if opened, else None. This is the single choke
        point where a BUY/SELL decision can actually become a position --
        everything upstream can only recommend, never execute.

        `sector` is read from `result.sector` (set by scan_symbol()) rather
        than taken as a separate parameter here -- there is no other way
        for a caller holding just a ScanResult to recover what sector it
        was scanned with, and a separate parameter previously left every
        real call site (main.py, dashboard.py) silently passing
        sector=None, meaning max_sector_exposure_pct could never actually
        register or block anything.
        """
        sector = result.sector
        if not result.filter_result.approved or result.filter_result.final_decision not in ("BUY", "SELL"):
            return None

        # This engine holds at most one open paper position per symbol at
        # a time (_trade_ids is structurally a symbol -> single trade_id
        # map). Re-signaling on an already-open symbol would make
        # PaperBroker silently merge the fill into the existing position
        # while this method created a SECOND, brand-new JournalEntry --
        # orphaning the first one forever (it would never receive
        # record_close()). Refuse instead; re-entry is allowed again once
        # manage_open_positions() genuinely closes the position.
        if result.symbol in self._trade_ids:
            logger.info(
                "Skipping execution for %s: a paper position is already open (trade_id=%s).",
                result.symbol, self._trade_ids[result.symbol],
            )
            return None

        # Recompute risk assessment fresh at execution time (price may have
        # moved) via the SAME pipeline.propose_stop_target /
        # pipeline.risk_calculator used by the initial scan and by
        # Backtester -- never a locally re-derived formula.
        current_price = self.market_data.get_quote(result.symbol)
        technical = self.technical_analyzer.analyze(result.symbol, self.market_data.get_daily(result.symbol))
        side = result.filter_result.final_decision
        stop, target = self.pipeline.propose_stop_target(side, current_price, technical.atr14)

        risk = self.pipeline.risk_calculator.evaluate(
            symbol=result.symbol, side=side, entry=current_price, stop_loss=stop,
            target=target, capital=self.broker.get_balance(),
            risk_multiplier=self.capital_protection.current_risk_multiplier(),
        )
        if not risk.approved:
            logger.info("Execution-time risk check failed for %s: %s", result.symbol, risk.rejection_reasons)
            return None

        order = self.broker.place_order(
            symbol=result.symbol, side=side, quantity=risk.position_size,
            stop_loss=risk.stop_loss, target=risk.target, sector=sector,
        )
        if order.status.value != "FILLED":
            logger.info("Paper order not filled for %s: %s", result.symbol, order.rejection_reason)
            return None

        # Real fee actually charged by PaperBroker for this entry leg
        # (config.risk.transaction_cost_pct is the exact same value the
        # broker was constructed with as fee_pct) -- must be recorded now,
        # not left at JournalEntry's fees=0.0 default, or this trade's
        # eventual net_pnl will omit it (see manage_open_positions).
        entry_fee = order.filled_price * order.filled_quantity * self.config.risk.transaction_cost_pct

        trade_id = str(uuid.uuid4())
        self._trade_ids[result.symbol] = trade_id
        self.journal.record_open(JournalEntry(
            trade_id=trade_id, symbol=result.symbol, side=side,
            entry_time=datetime.now(timezone.utc), entry_price=order.filled_price,
            stop_loss=risk.stop_loss, target=risk.target, quantity=order.filled_quantity,
            confidence_at_entry=result.signal.overall_confidence,
            decision_reasons="; ".join(result.filter_result.reasons or result.signal.reasons),
            fees=entry_fee,
            sector=sector,
        ))
        self.capital_protection.register_open(sector, risk.notional_exposure)
        return trade_id

    def manage_open_positions(self) -> None:
        """Checks every open paper position against its stop-loss/target
        using the latest quote and closes it if breached. Call this on each
        polling cycle (see main.py's `paper-trade` loop)."""
        for position in list(self.broker.get_positions()):
            try:
                quote = self.broker.get_quote(position.symbol)
            except Exception:
                continue

            hit_stop = (position.side == "LONG" and position.stop_loss and quote <= position.stop_loss) or \
                       (position.side == "SHORT" and position.stop_loss and quote >= position.stop_loss)
            hit_target = (position.side == "LONG" and position.target and quote >= position.target) or \
                         (position.side == "SHORT" and position.target and quote <= position.target)

            if not (hit_stop or hit_target):
                continue

            # Snapshot BEFORE calling place_order(): `position` here IS the
            # SAME object PaperBroker holds internally (get_positions()
            # returns the live Position instances, not copies). Closing it
            # mutates `position.quantity` in place (down to 0 for a full
            # close, which this call always requests) -- reading
            # position.quantity/average_price/sector AFTER place_order()
            # would silently compute gross_pnl as `x * 0 = 0` and the
            # sector-exposure notional as `x * 0 = 0` every time, exactly
            # the failure mode a direct test of this method caught.
            close_quantity = position.quantity
            close_average_price = position.average_price
            close_sector = position.sector
            close_side_label = position.side

            close_side = "SELL" if position.side == "LONG" else "BUY"
            order = self.broker.place_order(symbol=position.symbol, side=close_side, quantity=close_quantity)
            if order.status.value != "FILLED":
                continue

            trade_id = self._trade_ids.get(position.symbol)
            if trade_id:
                if close_side_label == "LONG":
                    gross = (order.filled_price - close_average_price) * close_quantity
                else:
                    gross = (close_average_price - order.filled_price) * close_quantity
                exit_fee = order.filled_price * order.filled_quantity * self.config.risk.transaction_cost_pct
                # The entry fee was recorded on this trade's JournalEntry
                # at open time (execute_if_approved) -- must be subtracted
                # here too, or net_pnl only ever reflects the exit leg's
                # cost and overstates every trade's profitability.
                entry_fee = next((e.fees for e in self.journal.open_trades() if e.trade_id == trade_id), 0.0)
                net = gross - entry_fee - exit_fee
                self.journal.record_close(
                    trade_id=trade_id, exit_time=datetime.now(timezone.utc), exit_price=order.filled_price,
                    exit_reason="TARGET" if hit_target else "STOP", fees=exit_fee, gross_pnl=gross, net_pnl=net,
                )
                self.capital_protection.register_close(close_sector, close_average_price * close_quantity, net)
                del self._trade_ids[position.symbol]

    def record_equity_snapshot(self) -> None:
        """Appends the current mark-to-market equity to this session's
        equity history, used by `account_summary()` to compute drawdown.
        PaperBroker itself has no memory of past equity -- only the engine
        knows "now" -- so this must be called periodically (every
        `account_summary()` call does it automatically; a polling loop like
        `main.py`'s `paper-trade` command can also call it directly on
        every cycle, e.g. inside `manage_open_positions()`)."""
        self._equity_history.append((datetime.now(timezone.utc), self.broker.equity()))

    def account_summary(self) -> dict:
        from backtesting.metrics import max_drawdown as _max_drawdown, sharpe_ratio as _sharpe_ratio

        self.record_equity_snapshot()
        equity = self.broker.equity()
        equity_values = [e for _, e in self._equity_history]
        peak_equity = max(equity_values) if equity_values else equity
        current_drawdown_pct = ((equity - peak_equity) / peak_equity) if peak_equity else 0.0
        max_drawdown_pct = _max_drawdown(equity_values) if equity_values else 0.0

        # Sharpe from THIS SESSION's own equity snapshots (spec section 18).
        # These snapshots land whenever account_summary()/record_equity_snapshot()
        # happen to be called -- irregular wall-clock intervals, not a clean
        # daily bar series -- so this is a rough, session-local estimate, not
        # a like-for-like comparison to a backtest's daily-bar Sharpe. Report
        # it as such; do not present it as more precise than it is.
        snapshot_returns = [
            (equity_values[i] - equity_values[i - 1]) / equity_values[i - 1]
            for i in range(1, len(equity_values)) if equity_values[i - 1]
        ]
        sharpe_ratio_session = _sharpe_ratio(snapshot_returns) if len(snapshot_returns) >= 2 else 0.0

        # Daily risk budget used (spec section 18: "Daily Risk Used") -- how
        # much of today's max-daily-loss allowance has been consumed by
        # REALIZED P&L so far today. A profitable day shows 0%, never negative.
        cp = self.capital_protection
        daily_loss_budget = cp.state.starting_capital * cp.max_daily_loss_pct
        daily_risk_used_pct = (
            max(0.0, -cp.state.daily_pnl) / daily_loss_budget if daily_loss_budget > 0 else 0.0
        )

        return {
            "starting_capital": self.broker.starting_capital,
            "cash": self.broker.get_balance(),
            "equity": equity,
            "unrealized_pnl": self.broker.unrealized_pnl(),
            "total_return_pct": (equity - self.broker.starting_capital) / self.broker.starting_capital,
            "open_positions": len(self.broker.get_positions()),
            "win_rate": self.journal.win_rate(),
            "profit_factor": self.journal.profit_factor(),
            "closed_trades": len(self.journal.closed_trades()),
            "trading_halted": self.capital_protection.state.trading_halted,
            "halt_reason": self.capital_protection.state.halt_reason,
            # Drawdown (spec sections 12, 17, 18) -- computed from THIS
            # session's own equity snapshot history, not fabricated or
            # backfilled from a shorter or different history.
            "current_drawdown_pct": current_drawdown_pct,
            "max_drawdown_pct": max_drawdown_pct,
            "equity_snapshots_recorded": len(self._equity_history),
            "sharpe_ratio_session": sharpe_ratio_session,
            "daily_risk_used_pct": daily_risk_used_pct,
            "no_trade_count": self.decision_log.no_trade_count(),
            "decisions_logged": len(self.decision_log.all_entries()),
        }
