"""
Small-account / affordability core (spec Parts 10-13, 18, 26, 31 partial).

Covers:
  * `affordable_quantity` (cash-only affordability) vs `position_size`
    (risk-budget- and notional-cap-bounded) as two DISTINCT reasons a trade
    can size to zero -- spec Part 10-11.
  * `small_account_risk_per_trade_pct` applying below the configured
    capital threshold, decided BEFORE any candidate is evaluated -- never
    adjusted after the fact to force a trade through (spec Part 11).
  * no fractional shares, ever.
  * `transaction_cost_pct_of_capital` reporting (spec Part 13).
  * `CapitalProtection` loss-streak risk reduction (spec Part 26).
  * candidate ranking by quality + affordability, not price alone (spec
    Part 12 -- mirrors the spec's own worked example).
  * the Part 18 decision-log schema populated end-to-end through
    `PaperTradingEngine.scan_symbol`.
  * `account-check` / `scan --capital` CLI argument wiring.

All tests run fully offline (synthetic data / injected fakes), consistent
with the rest of this suite (see tests/conftest.py).
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd
import pytest

from config.settings import Config
from data.market_data import MarketDataProvider
from fundamentals.fundamental_analysis import FundamentalAnalyzer
from paper_trading.candidate_ranking import affordable_candidates, rank_by_quality_and_affordability
from paper_trading.decision_log import DecisionLog
from paper_trading.engine import PaperTradingEngine, ScanResult
from paper_trading.journal import TradeJournal
from risk.risk_engine import CapitalProtection, TradeRiskCalculator
from strategy.report import TradeReport
from strategy.signal_engine import SignalDecision
from strategy.trade_filter import FilterResult
from tests.conftest import make_synthetic_ohlcv


# =============================================================================
# TradeRiskCalculator: affordability vs risk-budget sizing
# =============================================================================

def make_calculator(**overrides):
    defaults = dict(
        risk_per_trade_pct=0.01,
        min_risk_reward=2.0,
        transaction_cost_pct=0.001,
        slippage_pct=0.0007,
        min_edge_after_costs_pct=0.001,
        max_single_position_pct=0.20,
    )
    defaults.update(overrides)
    return TradeRiskCalculator(**defaults)


def test_no_capital_cannot_afford_even_one_share():
    """INR 50 account: even one share of a ~INR 100 stock is unaffordable
    by cash alone -- affordable_quantity and position_size must both be
    zero, with a distinct 'insufficient capital' reason, never a forced
    fractional or free share."""
    calc = make_calculator()
    result = calc.evaluate("TEST", "BUY", entry=100, stop_loss=97, target=108, capital=50)
    assert result.affordable_quantity == 0
    assert result.position_size == 0
    assert result.approved is False
    assert any("INSUFFICIENT CAPITAL" in r for r in result.rejection_reasons)


def test_affordable_by_cash_but_risk_per_share_too_large_for_budget():
    """INR 1,000 account (below the small-account threshold, so
    small_account_risk_per_trade_pct=0.02 applies -> max_risk_capital=20):
    a stock priced 100 with a 50-wide stop can easily be bought by cash
    (9 shares affordable) but ONE share alone risks 50, far above the
    20-rupee risk budget -- position_size must be 0 while affordable_
    quantity is > 0, with a rejection reason that says so distinctly (and
    NEVER silently raises the risk limit to force the trade)."""
    calc = make_calculator(transaction_cost_pct=0.001, slippage_pct=0.0007)
    result = calc.evaluate("TEST", "BUY", entry=100, stop_loss=50, target=200, capital=1_000)
    assert result.affordable_quantity >= 1
    assert result.position_size == 0
    assert result.approved is False
    assert any(
        "risk" in r.lower() and "budget" in r.lower() and "never secretly" in r.lower()
        for r in result.rejection_reasons
    )


def test_position_size_never_exceeds_affordable_quantity():
    calc = make_calculator(risk_per_trade_pct=0.5, max_single_position_pct=1.0)  # deliberately loose caps
    result = calc.evaluate("TEST", "BUY", entry=100, stop_loss=99, target=110, capital=1_000)
    assert result.position_size <= result.affordable_quantity


def test_quantities_are_always_whole_shares_never_fractional():
    calc = make_calculator()
    for capital in (100, 500, 1_000, 5_000, 25_000, 1_000_000):
        result = calc.evaluate("TEST", "BUY", entry=137.35, stop_loss=131.10, target=155.00, capital=capital)
        assert isinstance(result.position_size, int)
        assert isinstance(result.affordable_quantity, int)
        assert result.position_size >= 0
        assert result.affordable_quantity >= 0


def test_transaction_cost_pct_of_capital_is_reported():
    calc = make_calculator(risk_per_trade_pct=0.05, max_single_position_pct=1.0)
    result = calc.evaluate("TEST", "BUY", entry=100, stop_loss=95, target=115, capital=10_000)
    assert result.position_size > 0
    expected = (result.expected_transaction_cost + result.expected_slippage_cost) / 10_000
    assert result.transaction_cost_pct_of_capital == pytest.approx(expected)


# =============================================================================
# small_account_risk_per_trade_pct: applied by account size, not by outcome
# =============================================================================

def test_small_account_uses_the_small_account_risk_pct_below_threshold():
    """capital=1,000 < default threshold (25,000) -> effective risk pct is
    small_account_risk_per_trade_pct (0.02), NOT the base risk_per_trade_pct
    (0.01) -- verified by an exact-division setup where capital_at_risk ==
    max_risk_capital precisely (no notional/cash cap binding)."""
    calc = make_calculator(
        risk_per_trade_pct=0.01, small_account_risk_per_trade_pct=0.02,
        small_account_capital_threshold=25_000.0,
    )
    result = calc.evaluate("TEST", "BUY", entry=10, stop_loss=8, target=16, capital=1_000)
    assert result.capital_at_risk == pytest.approx(1_000 * 0.02)


def test_large_account_uses_the_base_risk_pct_at_or_above_threshold():
    calc = make_calculator(
        risk_per_trade_pct=0.01, small_account_risk_per_trade_pct=0.02,
        small_account_capital_threshold=25_000.0,
    )
    result = calc.evaluate("TEST", "BUY", entry=10, stop_loss=8, target=16, capital=1_000_000)
    assert result.capital_at_risk == pytest.approx(1_000_000 * 0.01)


def test_small_account_risk_pct_is_identical_regardless_of_whether_the_trade_would_pass():
    """The effective risk pct is fixed BEFORE any candidate-specific
    approval math runs -- a setup that will clearly be REJECTED (RR below
    minimum) must still show the exact same capital_at_risk formula as one
    that will be approved. This guards against 'raise risk until it works'
    regressions (spec Part 11)."""
    calc = make_calculator(
        risk_per_trade_pct=0.01, small_account_risk_per_trade_pct=0.02,
        min_risk_reward=2.0,
    )
    good = calc.evaluate("TEST", "BUY", entry=10, stop_loss=8, target=16, capital=1_000)  # RR=3, approved
    bad = calc.evaluate("TEST", "BUY", entry=10, stop_loss=8, target=10.5, capital=1_000)  # RR<2, rejected
    assert good.approved is True
    assert bad.approved is False
    assert good.capital_at_risk == pytest.approx(bad.capital_at_risk)
    assert good.capital_at_risk == pytest.approx(1_000 * 0.02)


# =============================================================================
# CapitalProtection: consecutive-loss risk reduction (spec Part 26)
# =============================================================================

def make_protection(**overrides):
    defaults = dict(
        starting_capital=1_000_000.0,
        max_daily_loss_pct=0.5,   # loose -- these tests aren't about the daily/weekly halt
        max_weekly_loss_pct=0.9,
        max_simultaneous_positions=10,
        max_sector_exposure_pct=0.9,
        max_single_position_pct=0.9,
        today=date(2024, 1, 2),
        max_consecutive_losses_before_reduction=3,
        risk_reduction_factor=0.5,
    )
    defaults.update(overrides)
    return CapitalProtection(**defaults)


def test_risk_multiplier_is_one_with_no_losses():
    protection = make_protection()
    assert protection.current_risk_multiplier() == pytest.approx(1.0)


def test_risk_multiplier_drops_after_streak_threshold():
    protection = make_protection(max_consecutive_losses_before_reduction=3, risk_reduction_factor=0.5)
    for _ in range(2):
        protection.register_close(sector=None, notional_exposure=1_000, realized_pnl=-100, today=date(2024, 1, 2))
    assert protection.current_risk_multiplier() == pytest.approx(1.0)  # only 2 losses so far
    protection.register_close(sector=None, notional_exposure=1_000, realized_pnl=-100, today=date(2024, 1, 2))
    assert protection.state.consecutive_losses == 3
    assert protection.current_risk_multiplier() == pytest.approx(0.5)


def test_a_win_resets_the_loss_streak():
    protection = make_protection(max_consecutive_losses_before_reduction=2, risk_reduction_factor=0.5)
    protection.register_close(sector=None, notional_exposure=1_000, realized_pnl=-100, today=date(2024, 1, 2))
    protection.register_close(sector=None, notional_exposure=1_000, realized_pnl=-100, today=date(2024, 1, 2))
    assert protection.current_risk_multiplier() == pytest.approx(0.5)
    protection.register_close(sector=None, notional_exposure=1_000, realized_pnl=+500, today=date(2024, 1, 2))
    assert protection.state.consecutive_losses == 0
    assert protection.current_risk_multiplier() == pytest.approx(1.0)


def test_risk_multiplier_reduces_position_size_but_never_increases_it():
    calc = make_calculator(risk_per_trade_pct=0.02, max_single_position_pct=1.0)
    full = calc.evaluate("TEST", "BUY", entry=50, stop_loss=48, target=56, capital=1_000_000, risk_multiplier=1.0)
    reduced = calc.evaluate("TEST", "BUY", entry=50, stop_loss=48, target=56, capital=1_000_000, risk_multiplier=0.5)
    assert reduced.position_size < full.position_size
    assert reduced.position_size == pytest.approx(full.position_size // 2, abs=1)


# =============================================================================
# Candidate ranking: quality + affordability, never price alone (spec Part 12)
# =============================================================================

def _fake_scan_result(symbol, price, decision, approved, affordable_quantity, position_size,
                       confidence, expected_value_total, risk_reward_ratio, reason=None, has_risk=True):
    signal = SignalDecision(
        symbol=symbol, component_scores={}, overall_confidence=confidence,
        confidence_label="HIGH" if confidence >= 70 else "NO TRADE", direction="UP" if decision == "BUY" else "NONE",
        model_agreement=1.0, decision=decision, reasons=[reason] if reason else [],
    )
    filter_result = FilterResult(
        symbol=symbol, approved=approved, final_decision=decision,
        checklist={}, reasons=[reason] if (reason and not approved) else [],
    )
    report = TradeReport(
        symbol=symbol, current_price=price, market_trend="UPTREND", sector_trend="N/A",
        technical_score=70.0, fundamental_score=None, news_score=None, social_score=None, risk_score=60.0,
        overall_confidence=confidence, confidence_label=signal.confidence_label,
        entry=price, stop_loss=price * 0.95, target=price * 1.15, risk_reward=risk_reward_ratio,
        expected_risk=None, expected_reward=None, decision=decision, reasons=[],
        invalidation_condition="test",
    )

    class _FakeRisk:
        pass

    risk = None
    if has_risk:
        # A RiskAssessment exists whenever the signal engine reached a
        # BUY/SELL direction at all -- REGARDLESS of whether the resulting
        # affordable_quantity/position_size came out to zero. Only a symbol
        # that never reached a direction this cycle has risk=None (see
        # paper_trading/candidate_ranking.py's affordability_basis).
        risk = _FakeRisk()
        risk.position_size = position_size
        risk.affordable_quantity = affordable_quantity
        risk.expected_value_total = expected_value_total
        risk.risk_reward_ratio = risk_reward_ratio
        risk.expected_transaction_cost = 0.0
        risk.expected_slippage_cost = 0.0

    return ScanResult(symbol=symbol, report=report, signal=signal, filter_result=filter_result, risk=risk)


def test_ranking_prefers_affordable_positive_ev_setup_over_expensive_negative_ev_one():
    """Mirrors the spec's own worked example (Part 32) for a ~INR 1,000
    account: Stock A (INR 850, 1 share WAS priced by the risk engine, but
    negative EV) must rank BELOW Stock B (INR 180, affordable, positive EV)
    -- price alone must never decide the ranking. Stock C is affordable but
    a weak setup, and must still rank behind the genuinely good trade."""
    stock_a = _fake_scan_result(  # spec Part 32 "Stock A"
        "STOCKA", price=850, decision="NO TRADE", approved=False,
        affordable_quantity=1, position_size=1, confidence=55.0,
        expected_value_total=-12.0, risk_reward_ratio=1.2, reason="Negative expected value.",
    )
    stock_b = _fake_scan_result(  # spec Part 32 "Stock B"
        "STOCKB", price=180, decision="BUY", approved=True,
        affordable_quantity=5, position_size=1, confidence=76.0,
        expected_value_total=18.0, risk_reward_ratio=3.1,
    )
    stock_c = _fake_scan_result(  # spec Part 32 "Stock C"
        "STOCKC", price=50, decision="NO TRADE", approved=False,
        affordable_quantity=10, position_size=10, confidence=40.0,
        expected_value_total=-5.0, risk_reward_ratio=0.8, reason="Weak setup / poor liquidity.",
    )
    ranked = rank_by_quality_and_affordability([stock_a, stock_b, stock_c], capital=1_000.0)
    assert ranked[0].symbol == "STOCKB"
    affordable = affordable_candidates(ranked)
    assert {c.symbol for c in affordable} == {"STOCKA", "STOCKB", "STOCKC"}  # all priced by risk engine
    assert all(c.affordability_basis == "risk_assessment" for c in ranked)


def test_genuinely_unaffordable_never_outranks_affordable_approved_purely_on_confidence():
    """A candidate the RISK ENGINE actually evaluated and found could not
    afford even one share (affordable_quantity=0, via a real RiskAssessment,
    NOT the 'no direction reached' fallback) must never outrank an
    affordable, approved candidate -- regardless of how high its raw
    confidence score is."""
    truly_unaffordable = _fake_scan_result(
        "BROKE", price=10, decision="NO TRADE", approved=False,
        affordable_quantity=0, position_size=0, confidence=95.0,
        expected_value_total=None, risk_reward_ratio=None, reason="INSUFFICIENT CAPITAL",
    )
    modest = _fake_scan_result(
        "MODEST", price=200, decision="BUY", approved=True,
        affordable_quantity=2, position_size=1, confidence=71.0,
        expected_value_total=5.0, risk_reward_ratio=2.2,
    )
    ranked = rank_by_quality_and_affordability([truly_unaffordable, modest], capital=1_000.0)
    assert ranked[0].symbol == "MODEST"
    assert {c.symbol for c in affordable_candidates(ranked)} == {"MODEST"}


# --- Regression: "no directional signal reached" must NEVER be reported as
# "insufficient capital" -- caught via manual CLI testing where every
# scanned symbol correctly returned NO TRADE (no direction reached, so
# risk=None) and the account-check report incorrectly said the account
# couldn't afford anything, even for cheap, clearly-affordable stocks. ------

def test_no_direction_reached_falls_back_to_price_only_affordability_estimate():
    no_signal = _fake_scan_result(
        "QUIETCO", price=170, decision="NO TRADE", approved=False,
        affordable_quantity=0, position_size=0, confidence=63.0,  # confidence too low, no direction
        expected_value_total=None, risk_reward_ratio=None,
        reason="Confidence 63.0 below minimum required 70.0.", has_risk=False,
    )
    ranked = rank_by_quality_and_affordability([no_signal], capital=1_000.0)
    candidate = ranked[0]
    assert candidate.affordability_basis == "price_only"
    assert candidate.affordable_quantity == 5  # int(1000 // 170) -- NOT zero
    assert candidate in affordable_candidates(ranked)


def test_price_only_estimate_is_genuinely_zero_when_price_exceeds_capital():
    no_signal = _fake_scan_result(
        "PRICEY", price=5_000, decision="NO TRADE", approved=False,
        affordable_quantity=0, position_size=0, confidence=50.0,
        expected_value_total=None, risk_reward_ratio=None, has_risk=False,
    )
    ranked = rank_by_quality_and_affordability([no_signal], capital=1_000.0)
    assert ranked[0].affordability_basis == "price_only"
    assert ranked[0].affordable_quantity == 0
    assert ranked[0] not in affordable_candidates(ranked)


# =============================================================================
# Decision log: Part 18 schema populated end-to-end
# =============================================================================

def _recent_start_date(n: int) -> str:
    """Data quality's staleness check (data/data_quality.py) compares the
    last bar to REAL wall-clock 'now' -- conftest.py's default
    start_date="2023-01-02" is far in the past by now and would get these
    decisions gated to NO TRADE by the data-quality check before ever
    reaching the signal engine. Anchor synthetic data to end near today
    instead, so these tests exercise the signal-scoring/decision-log path
    they're actually about."""
    return pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=n)[0].strftime("%Y-%m-%d")


def make_offline_engine(daily_df, tmp_path, config=None, starting_capital=None) -> PaperTradingEngine:
    config = config or Config()
    config.decision_thresholds.min_history_bars = 60
    if starting_capital is not None:
        config.paper_trading.starting_capital = starting_capital

    def fetch(symbol, period, interval):
        return daily_df.copy()

    md = MarketDataProvider(fetch_fn=fetch)

    def fake_fundamentals_fetch(symbol):
        return {
            "revenue_growth": 0.1, "earnings_growth": 0.1, "eps": 5, "pe_ratio": 20, "pb_ratio": 3,
            "debt_to_equity": 50, "roe": 0.15, "profit_margin": 0.1, "operating_cash_flow": 100_000,
            "free_cash_flow": 50_000, "last_update": datetime.now(timezone.utc),
        }

    return PaperTradingEngine(
        config=config, market_data=md, fetch_news=False, fetch_social=False,
        fundamental_analyzer=FundamentalAnalyzer(fetch_fn=fake_fundamentals_fetch),
        journal=TradeJournal(path=str(tmp_path / "journal.csv")),
        decision_log=DecisionLog(path=str(tmp_path / "decisions.csv")),
    )


def test_decision_log_entry_carries_the_part18_schema(tmp_path):
    daily = make_synthetic_ohlcv(n=260, start_price=150.0, drift=0.001, seed=7, start_date=_recent_start_date(260))
    engine = make_offline_engine(daily, tmp_path, starting_capital=50_000.0)
    engine.scan_symbol("SMALLCO")
    entries = engine.decision_log.all_entries()
    assert len(entries) == 1
    entry = entries[0]

    # Always computable for a scan that reaches the signal engine:
    assert entry.account_balance == pytest.approx(50_000.0)
    assert entry.price is not None and entry.price > 0
    assert entry.technical_score is not None
    assert entry.risk_score is not None
    assert entry.rejection_reason is not None or entry.approved

    # sector_score: no sector-index data source exists anywhere in this
    # system -- honestly None, never fabricated.
    assert entry.sector_score is None
    # momentum_score: genuinely computable once technical analysis ran for
    # this decision (indicators/technical.py's momentum_score()) -- must
    # be a real, bounded value here, not None.
    assert entry.momentum_score is not None
    assert 0 <= entry.momentum_score <= 100


def test_decision_log_records_risk_fields_when_a_direction_is_reached(tmp_path):
    # A strong, low-noise uptrend is likely (not guaranteed) to reach BUY;
    # either way, if `risk` was computed, its fields must be threaded through.
    daily = make_synthetic_ohlcv(
        n=260, start_price=150.0, drift=0.006, volatility=0.006, seed=11, start_date=_recent_start_date(260),
    )
    engine = make_offline_engine(daily, tmp_path, starting_capital=1_000_000.0)
    result = engine.scan_symbol("BIGCO")
    entry = engine.decision_log.all_entries()[0]
    if result.risk is not None:
        assert entry.position_size == result.risk.position_size
        assert entry.affordable_quantity == result.risk.affordable_quantity
        assert entry.risk_reward_ratio == pytest.approx(result.risk.risk_reward_ratio)
        assert entry.expected_value == pytest.approx(result.risk.expected_value_total)


# =============================================================================
# Config validation
# =============================================================================

def test_config_has_small_account_defaults():
    cfg = Config()
    assert cfg.risk.small_account_capital_threshold == pytest.approx(25_000.0)
    assert cfg.risk.small_account_risk_per_trade_pct == pytest.approx(0.02)
    assert cfg.risk.max_consecutive_losses_before_reduction == 3
    assert cfg.risk.risk_reduction_factor == pytest.approx(0.5)
    cfg.validate()  # must not raise


def test_config_rejects_unsafe_small_account_risk_pct():
    cfg = Config()
    cfg.risk.small_account_risk_per_trade_pct = 0.5  # 50% -- unsafe even for a tiny account
    with pytest.raises(ValueError):
        cfg.validate()


def test_config_rejects_invalid_risk_reduction_factor():
    cfg = Config()
    cfg.risk.risk_reduction_factor = 1.5
    with pytest.raises(ValueError):
        cfg.validate()


# =============================================================================
# CLI argument wiring (argparse-level only -- no network calls)
# =============================================================================

def test_cli_scan_accepts_capital_override():
    import main
    parser = main.build_parser()
    args = parser.parse_args(["scan", "TEST.NS", "--capital", "500"])
    assert args.capital == pytest.approx(500.0)
    assert args.func is main.cmd_scan


def test_cli_scan_capital_defaults_to_none():
    import main
    parser = main.build_parser()
    args = parser.parse_args(["scan", "TEST.NS"])
    assert args.capital is None


def test_cli_account_check_requires_capital_and_registers_handler():
    import main
    parser = main.build_parser()
    args = parser.parse_args(["account-check", "--capital", "1000"])
    assert args.capital == pytest.approx(1_000.0)
    assert args.func is main.cmd_account_check

    with pytest.raises(SystemExit):
        parser.parse_args(["account-check"])  # --capital is required
