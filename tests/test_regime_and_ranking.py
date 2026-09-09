"""
Phase 2: market-regime detection (relocated + wired into live scanning) and
cross-sectional ranking (spec Part 5.5-5.6, Part 16).

Covers:
  * `classify_regime` relocation from backtesting/backtester.py to
    data/market_data.py is behavior-preserving (both import paths resolve
    to the same logic; `i=None` == "classify as of the last bar").
  * Spec Part 16: a BEAR regime raises SignalEngine's effective confidence
    threshold, and ONLY a BEAR regime -- never lowers it, never changes the
    underlying score.
  * `market_regime` is threaded end-to-end into SignalDecision/TradeReport/
    DecisionLogEntry during a live scan.
  * Cross-sectional technical-score percentile ranking within one scan
    cycle (paper_trading/candidate_ranking.py).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from config.settings import Config
from data.market_data import MarketDataProvider, classify_regime
from fundamentals.fundamental_analysis import FundamentalAnalyzer
from paper_trading.candidate_ranking import attach_cross_sectional_ranks, rank_by_quality_and_affordability
from paper_trading.decision_log import DecisionLog
from paper_trading.engine import PaperTradingEngine, ScanResult
from paper_trading.journal import TradeJournal
from strategy.report import TradeReport
from strategy.signal_engine import SignalDecision, SignalInputs, SignalEngine
from strategy.trade_filter import FilterResult
from tests.conftest import make_synthetic_ohlcv, make_uptrend_ohlcv


# =============================================================================
# classify_regime relocation
# =============================================================================

def test_classify_regime_importable_from_both_locations(uptrend_daily):
    from backtesting.backtester import classify_regime as from_backtester
    from data.market_data import classify_regime as from_market_data
    assert from_backtester is from_market_data  # same function object -- true relocation, not a copy

    i = len(uptrend_daily) - 1
    assert from_backtester(uptrend_daily, i) == from_market_data(uptrend_daily, i)


def test_classify_regime_i_none_means_last_bar(uptrend_daily):
    assert classify_regime(uptrend_daily, i=None) == classify_regime(uptrend_daily, i=len(uptrend_daily) - 1)


def test_classify_regime_labels_uptrend_bull(uptrend_daily):
    assert classify_regime(uptrend_daily).startswith("BULL")


def test_classify_regime_unknown_with_insufficient_history():
    tiny = make_synthetic_ohlcv(n=10)
    assert classify_regime(tiny) == "UNKNOWN"


# =============================================================================
# SignalEngine: bearish-regime confidence bump (spec Part 16)
# =============================================================================

class _FakeTechnical:
    """Duck-typed stand-in for indicators.technical.TechnicalSnapshot --
    only the attributes SignalEngine.decide() actually reads, with a fixed
    score so `overall` is fully deterministic (see weights below: only
    `technical` has nonzero weight, so overall == this value exactly,
    regardless of what volume/risk context scores compute to).
    SignalEngine now derives "technical" from trend_score()/momentum_score()/
    volatility_volume_score() (Phase 10) rather than technical_score()
    directly -- all three (and technical_score() itself, kept for any other
    caller) return the SAME fixed value here, so this fixture's "overall ==
    score exactly" contract still holds."""

    def __init__(self, score: float):
        self._score = score
        self.trend = "UPTREND"
        self.obv_trend = "FLAT"
        self.atr14 = 1.0

    def trend_score(self):
        return self._score

    def momentum_score(self):
        return self._score

    def volatility_volume_score(self):
        return self._score

    def technical_score(self):
        return self._score


def _technical_only_engine(**threshold_overrides):
    cfg = Config()
    cfg.signal_weights.technical = 1.0
    cfg.signal_weights.market_condition = 0.0
    cfg.signal_weights.fundamentals = 0.0
    cfg.signal_weights.news_sentiment = 0.0
    cfg.signal_weights.social_sentiment = 0.0
    cfg.signal_weights.volume_price_behavior = 0.0
    cfg.signal_weights.risk_volatility = 0.0
    cfg.decision_thresholds.min_confidence_to_trade = 70.0
    cfg.decision_thresholds.bearish_regime_confidence_bonus = 5.0
    cfg.decision_thresholds.min_independent_signals = 1
    cfg.decision_thresholds.min_history_bars = 1
    for k, v in threshold_overrides.items():
        setattr(cfg.decision_thresholds, k, v)
    return SignalEngine(cfg.signal_weights, cfg.decision_thresholds, cfg.confidence_bands)


def _make_inputs(score: float, market_regime: str) -> SignalInputs:
    return SignalInputs(
        symbol="TEST", technical=_FakeTechnical(score), market_trend="UNKNOWN", relative_strength=float("nan"),
        fundamentals=None, news=None, social=None, avg_volume_20d=1_000_000, min_liquidity_avg_volume=100_000,
        atr_pct_of_price=0.02, max_atr_pct_of_price=0.08, min_atr_pct_of_price=0.002, history_bars=300,
        market_regime=market_regime,
    )


def test_overall_confidence_is_identical_regardless_of_regime():
    """Regime must change the THRESHOLD, never the score itself."""
    engine = _technical_only_engine()
    sideways = engine.decide(_make_inputs(73.0, "SIDEWAYS_LOW_VOL"))
    bear = engine.decide(_make_inputs(73.0, "BEAR_LOW_VOL"))
    assert sideways.overall_confidence == pytest.approx(bear.overall_confidence)
    assert sideways.overall_confidence == pytest.approx(73.0)


def test_bear_regime_blocks_a_trade_that_would_otherwise_pass():
    """confidence=73 clears min_confidence_to_trade=70 but NOT the BEAR-
    bumped 75 -- the exact scenario spec Part 16 asks for."""
    engine = _technical_only_engine()
    non_bear = engine.decide(_make_inputs(73.0, "SIDEWAYS_LOW_VOL"))
    bear = engine.decide(_make_inputs(73.0, "BEAR_LOW_VOL"))
    assert non_bear.decision == "BUY"
    assert bear.decision != "BUY"
    assert any("BEAR" in r or "raised" in r.lower() for r in bear.reasons)


def test_non_bear_regimes_never_lower_the_threshold():
    """A confidence just BELOW the base threshold must still fail under
    every regime label, including UNKNOWN -- no regime can make trading
    easier, only a BEAR one can make it harder."""
    engine = _technical_only_engine()
    for regime in ("SIDEWAYS_LOW_VOL", "BULL_HIGH_VOL", "UNKNOWN", "BULL_LOW_VOL"):
        result = engine.decide(_make_inputs(69.0, regime))
        assert result.decision != "BUY", f"regime={regime} unexpectedly allowed a sub-threshold trade"


def test_market_regime_is_threaded_onto_the_signal_decision():
    engine = _technical_only_engine()
    result = engine.decide(_make_inputs(80.0, "BULL_LOW_VOL"))
    assert result.market_regime == "BULL_LOW_VOL"


# =============================================================================
# Live-path wiring: market_regime reaches the decision log / report
# =============================================================================

def _recent_start_date(n: int) -> str:
    return pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=n)[0].strftime("%Y-%m-%d")


def make_offline_engine(daily_df, tmp_path, config=None) -> PaperTradingEngine:
    config = config or Config()
    config.decision_thresholds.min_history_bars = 60

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


def test_scan_symbol_computes_a_real_market_regime_not_unknown(tmp_path):
    daily = make_synthetic_ohlcv(
        n=260, start_price=150.0, drift=0.004, volatility=0.008, seed=21, start_date=_recent_start_date(260),
    )
    engine = make_offline_engine(daily, tmp_path)
    result = engine.scan_symbol("REGIMECO")
    assert result is not None
    assert result.signal.market_regime != "UNKNOWN"
    assert result.report.market_regime == result.signal.market_regime


def test_decision_log_records_the_market_regime(tmp_path):
    daily = make_synthetic_ohlcv(
        n=260, start_price=150.0, drift=0.004, volatility=0.008, seed=22, start_date=_recent_start_date(260),
    )
    engine = make_offline_engine(daily, tmp_path)
    engine.scan_symbol("REGIMECO2")
    entry = engine.decision_log.all_entries()[0]
    assert entry.market_regime != "UNKNOWN"


# =============================================================================
# Cross-sectional ranking (spec Part 5.5)
# =============================================================================

def _fake_result_with_technical_score(symbol, technical_score):
    signal = SignalDecision(
        symbol=symbol, component_scores={"technical": technical_score} if technical_score is not None else {},
        overall_confidence=50.0, confidence_label="WATCH", direction="NONE", model_agreement=1.0,
        decision="NO TRADE", reasons=[],
    )
    filter_result = FilterResult(symbol=symbol, approved=False, final_decision="NO TRADE", checklist={}, reasons=[])
    report = TradeReport(
        symbol=symbol, current_price=100.0, market_trend="UPTREND", sector_trend="N/A",
        technical_score=technical_score, fundamental_score=None, news_score=None, social_score=None,
        risk_score=None, overall_confidence=50.0, confidence_label="WATCH", entry=None, stop_loss=None,
        target=None, risk_reward=None, expected_risk=None, expected_reward=None, decision="NO TRADE", reasons=[],
        invalidation_condition="test",
    )
    return ScanResult(symbol=symbol, report=report, signal=signal, filter_result=filter_result, risk=None)


def test_cross_sectional_percentile_ranks_relative_to_the_scanned_batch():
    results = [
        _fake_result_with_technical_score("LOW", 30.0),
        _fake_result_with_technical_score("MID", 60.0),
        _fake_result_with_technical_score("HIGH", 90.0),
    ]
    ranked = rank_by_quality_and_affordability(results, capital=100_000.0)
    by_symbol = {c.symbol: c for c in ranked}
    assert by_symbol["HIGH"].technical_percentile == pytest.approx(100.0)
    assert by_symbol["LOW"].technical_percentile == pytest.approx(33.3, abs=0.1)
    assert by_symbol["MID"].technical_percentile == pytest.approx(66.7, abs=0.1)


def test_cross_sectional_ranking_never_used_as_a_hard_gate():
    """Percentile is purely informational -- even the WORST-ranked
    candidate in a batch keeps its own independently-computed
    approved/decision fields untouched."""
    results = [
        _fake_result_with_technical_score("LOW", 30.0),
        _fake_result_with_technical_score("HIGH", 90.0),
    ]
    ranked = rank_by_quality_and_affordability(results, capital=100_000.0)
    for c in ranked:
        assert c.approved is False  # both were built as NO TRADE -- ranking doesn't change that
        assert c.decision == "NO TRADE"


def test_percentile_is_none_with_fewer_than_two_scored_candidates():
    results = [_fake_result_with_technical_score("ONLY", 50.0)]
    ranked = rank_by_quality_and_affordability(results, capital=100_000.0)
    assert ranked[0].technical_percentile is None


def test_percentile_skips_candidates_with_no_technical_score():
    results = [
        _fake_result_with_technical_score("NOSIGNAL", None),
        _fake_result_with_technical_score("LOW", 30.0),
        _fake_result_with_technical_score("HIGH", 90.0),
    ]
    ranked = rank_by_quality_and_affordability(results, capital=100_000.0)
    by_symbol = {c.symbol: c for c in ranked}
    assert by_symbol["NOSIGNAL"].technical_percentile is None
    # The two real scores are still ranked against EACH OTHER, not diluted
    # by the missing one.
    assert by_symbol["HIGH"].technical_percentile == pytest.approx(100.0)
    assert by_symbol["LOW"].technical_percentile == pytest.approx(50.0)


def test_attach_cross_sectional_ranks_is_idempotent():
    results = [
        _fake_result_with_technical_score("A", 40.0),
        _fake_result_with_technical_score("B", 80.0),
    ]
    ranked = rank_by_quality_and_affordability(results, capital=100_000.0)
    before = {c.symbol: c.technical_percentile for c in ranked}
    attach_cross_sectional_ranks(ranked)
    after = {c.symbol: c.technical_percentile for c in ranked}
    assert before == after
