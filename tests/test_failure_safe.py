"""
Spec section 22 (fail-safe) + section 3 (data-quality gate), tested at the
level that matters: end-to-end through StrategyPipeline, Backtester, and
PaperTradingEngine -- not just the DataQualityChecker in isolation
(see tests/test_data_quality.py for that).
"""

import pandas as pd
import pytest

from backtesting.backtester import Backtester
from config.settings import Config
from data.data_quality import DataQualityChecker, DataQualityIssue, DataQualityReport
from data.market_calendar import get_calendar
from data.market_data import MarketDataProvider
from fundamentals.fundamental_analysis import FundamentalSnapshot
from indicators.technical import TechnicalSnapshot
from news.news_analysis import NewsAggregate
from paper_trading.engine import PaperTradingEngine
from paper_trading.journal import TradeJournal
from sentiment.social_sentiment import SocialAggregate
from strategy.pipeline import build_pipeline
from strategy.signal_engine import SignalInputs


def make_config(**overrides):
    config = Config()
    config.decision_thresholds.min_history_bars = 60
    config.decision_thresholds.min_independent_signals = 1
    for k, v in overrides.items():
        if hasattr(config.decision_thresholds, k):
            setattr(config.decision_thresholds, k, v)
    return config


def make_bullish_inputs() -> SignalInputs:
    technical = TechnicalSnapshot(
        symbol="X", last_close=110, sma20=105, sma50=100, sma200=90, rsi14=58,
        macd_hist=1.2, macd_bullish_cross=True, atr14=2.0, atr_pct_of_price=0.018,
        bb_upper=115, bb_lower=95, bb_percent_b=0.6, obv_trend="RISING", trend="UPTREND",
    )
    return SignalInputs(
        symbol="X", technical=technical, market_trend="UPTREND", relative_strength=0.05,
        fundamentals=FundamentalSnapshot(
            symbol="X", revenue_growth=0.2, earnings_growth=0.2, eps=5, pe_ratio=18, pb_ratio=3,
            debt_to_equity=40, roe=0.2, profit_margin=0.15, operating_cash_flow=1_000_000,
            free_cash_flow=500_000, last_update=None, data_quality=1.0, is_stale=False,
        ),
        news=NewsAggregate(symbol="X", items=[], overall_sentiment_score=70, overall_confidence=90,
                            contradictory=False, num_credible_sources=3),
        social=SocialAggregate(symbol="X", posts=[], mention_volume=50, sentiment_mean=0.6,
                                positive_ratio=0.8, negative_ratio=0.05, sentiment_acceleration=0.1,
                                abnormal_activity=False, bot_like_fraction=0.05,
                                manipulation_suspected=False, credibility_factor=0.7, data_available=True),
        avg_volume_20d=1_000_000, min_liquidity_avg_volume=100_000, atr_pct_of_price=0.018,
        max_atr_pct_of_price=0.08, min_atr_pct_of_price=0.002, history_bars=200,
    )


def bad_quality_report(symbol="X") -> DataQualityReport:
    return DataQualityReport(
        symbol=symbol, as_of=pd.Timestamp.now(tz="UTC"),
        issues=[DataQualityIssue("invalid_ohlc", "CRITICAL", "simulated corrupted data for a test")],
        quality_score=0.1, status="INVALID", min_quality_score_to_trade=0.70,
    )


# --- StrategyPipeline: data quality gate overrides an otherwise-bullish signal --

def test_pipeline_forces_no_trade_when_data_quality_invalid_even_with_bullish_inputs():
    config = make_config()
    pipeline = build_pipeline(config)
    inputs = make_bullish_inputs()

    without_gate = pipeline.decide(
        inputs, current_price=110, capital=1_000_000, account_check_fn=lambda r: [], market_trend="UPTREND",
    )
    assert without_gate.filter_result.final_decision in ("BUY", "SELL", "HOLD", "NO TRADE")  # sanity: runs at all

    with_bad_quality = pipeline.decide(
        inputs, current_price=110, capital=1_000_000, account_check_fn=lambda r: [], market_trend="UPTREND",
        data_quality=bad_quality_report(),
    )
    assert with_bad_quality.filter_result.final_decision == "NO TRADE"
    assert with_bad_quality.filter_result.approved is False
    assert with_bad_quality.risk is None
    assert any("Data quality gate failed" in r for r in with_bad_quality.signal.reasons)


def test_pipeline_proceeds_normally_when_data_quality_is_fine():
    config = make_config()
    pipeline = build_pipeline(config)
    inputs = make_bullish_inputs()
    good_quality = DataQualityReport(
        symbol="X", as_of=pd.Timestamp.now(tz="UTC"), issues=[], quality_score=1.0,
        status="OK", min_quality_score_to_trade=0.70,
    )
    result = pipeline.decide(
        inputs, current_price=110, capital=1_000_000, account_check_fn=lambda r: [], market_trend="UPTREND",
        data_quality=good_quality,
    )
    # Not gated -- whatever it decides is decided on the merits, not forced NO TRADE by data quality.
    assert not any("Data quality gate failed" in r for r in result.signal.reasons)


# --- Backtester: a run-level data-quality failure invalidates the WHOLE run --

def test_backtester_marks_run_invalid_and_takes_no_trades_on_corrupted_data(uptrend_daily):
    config = make_config()
    corrupted = pd.concat([uptrend_daily.iloc[:5], uptrend_daily.iloc[2:3], uptrend_daily.iloc[5:]])  # duplicate timestamp
    bt = Backtester(config=config)
    result = bt.run("TEST", corrupted)
    assert result.is_valid is False
    assert result.data_quality is not None
    assert result.data_quality.status == "INVALID"
    assert result.trades == []
    assert result.metrics.num_trades == 0


def test_backtester_marks_run_valid_on_clean_data_and_reports_quality(uptrend_daily):
    config = make_config()
    bt = Backtester(config=config)
    result = bt.run("TEST", uptrend_daily)
    assert result.is_valid is True
    assert result.data_quality is not None
    assert result.data_quality.symbol == "TEST"


# --- PaperTradingEngine: provider failure never becomes a trade -----------

def make_fetch_fn(df):
    def fetch(symbol, period, interval):
        return df.copy()
    return fetch


def test_scan_symbol_never_trades_on_badly_corrupted_data(uptrend_daily, tmp_path):
    config = Config()
    config.decision_thresholds.min_history_bars = 60
    corrupted = uptrend_daily.copy()
    corrupted.iloc[10, corrupted.columns.get_loc("High")] = corrupted.iloc[10]["Low"] - 5  # invalid OHLC
    md = MarketDataProvider(fetch_fn=make_fetch_fn(corrupted))
    engine = PaperTradingEngine(
        config=config, market_data=md, fetch_news=False, fetch_social=False,
        journal=TradeJournal(path=str(tmp_path / "journal.csv")),
    )
    result = engine.scan_symbol("FAKE")
    assert result is not None
    assert result.data_quality is not None
    assert result.data_quality.has_critical_issues is True
    assert result.filter_result.final_decision == "NO TRADE"


def test_scan_symbol_returns_none_not_a_crash_when_data_totally_unavailable(tmp_path):
    from data.market_data import DataUnavailableError

    def always_fails(symbol, period, interval):
        raise DataUnavailableError("simulated total provider outage")

    config = Config()
    md = MarketDataProvider(fetch_fn=always_fails, max_retries=1)
    engine = PaperTradingEngine(
        config=config, market_data=md, fetch_news=False, fetch_social=False,
        journal=TradeJournal(path=str(tmp_path / "journal.csv")),
    )
    result = engine.scan_symbol("FAKE")
    assert result is None  # never trades; never raises
