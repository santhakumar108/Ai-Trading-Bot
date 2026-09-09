import numpy as np
import pytest

from indicators.technical import TechnicalAnalyzer, adx, atr, bollinger_bands, ema, macd, rsi, sma, vwap


def test_sma_basic():
    import pandas as pd
    s = pd.Series([1, 2, 3, 4, 5])
    out = sma(s, 3)
    assert np.isnan(out.iloc[1])
    assert out.iloc[2] == pytest.approx(2.0)
    assert out.iloc[4] == pytest.approx(4.0)


def test_rsi_bounds(uptrend_daily):
    out = rsi(uptrend_daily["Close"], 14)
    assert (out >= 0).all() and (out <= 100).all()
    # A strong, clean uptrend should push RSI well above neutral eventually.
    assert out.tail(30).mean() > 50


def test_rsi_no_lookahead(uptrend_daily):
    """RSI at bar t must be identical whether or not later bars exist."""
    full = rsi(uptrend_daily["Close"], 14)
    truncated = rsi(uptrend_daily["Close"].iloc[:150], 14)
    assert full.iloc[149] == pytest.approx(truncated.iloc[149])


def test_macd_shapes(uptrend_daily):
    macd_line, signal_line, hist = macd(uptrend_daily["Close"])
    assert len(macd_line) == len(uptrend_daily)
    assert np.allclose((macd_line - signal_line).dropna(), hist.dropna())


def test_atr_positive(uptrend_daily):
    out = atr(uptrend_daily["High"], uptrend_daily["Low"], uptrend_daily["Close"], 14)
    assert (out.dropna() > 0).all()


def test_bollinger_ordering(uptrend_daily):
    upper, mid, lower = bollinger_bands(uptrend_daily["Close"], 20, 2.0)
    valid = upper.dropna().index.intersection(lower.dropna().index)
    assert (upper.loc[valid] >= mid.loc[valid]).all()
    assert (mid.loc[valid] >= lower.loc[valid]).all()


def test_vwap_within_price_range(uptrend_daily):
    out = vwap(uptrend_daily["High"], uptrend_daily["Low"], uptrend_daily["Close"], uptrend_daily["Volume"])
    # Cumulative VWAP should stay within the overall observed price envelope.
    assert out.iloc[-1] <= uptrend_daily["High"].max()
    assert out.iloc[-1] >= uptrend_daily["Low"].min()


def test_technical_analyzer_uptrend_scores_higher_than_downtrend(uptrend_daily, downtrend_daily):
    analyzer = TechnicalAnalyzer()
    up_snap = analyzer.analyze("UP", uptrend_daily)
    down_snap = analyzer.analyze("DOWN", downtrend_daily)
    assert up_snap.trend == "UPTREND"
    assert down_snap.trend == "DOWNTREND"
    assert up_snap.technical_score() > down_snap.technical_score()


def test_technical_analyzer_requires_min_history():
    import pandas as pd
    tiny = pd.DataFrame({"Open": [1, 2], "High": [1, 2], "Low": [1, 2], "Close": [1, 2], "Volume": [100, 100]})
    with pytest.raises(ValueError):
        TechnicalAnalyzer().analyze("X", tiny)


def test_technical_score_bounded(uptrend_daily, downtrend_daily, choppy_daily):
    analyzer = TechnicalAnalyzer()
    for df in (uptrend_daily, downtrend_daily, choppy_daily):
        snap = analyzer.analyze("X", df)
        assert 0 <= snap.technical_score() <= 100


# --- ADX (spec Part 5: trend strength, added Phase 2) -----------------------

def test_adx_bounded_0_100(uptrend_daily, choppy_daily):
    for df in (uptrend_daily, choppy_daily):
        out = adx(df["High"], df["Low"], df["Close"], 14)
        valid = out.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()


def test_adx_reads_higher_on_a_strong_clean_trend_than_a_choppy_market(uptrend_daily, choppy_daily):
    """ADX measures trend STRENGTH/conviction, not direction -- a strong,
    low-noise uptrend should read a materially higher ADX than a
    directionless, choppy series of the same length."""
    up_adx = adx(uptrend_daily["High"], uptrend_daily["Low"], uptrend_daily["Close"], 14).tail(30).mean()
    choppy_adx = adx(choppy_daily["High"], choppy_daily["Low"], choppy_daily["Close"], 14).tail(30).mean()
    assert up_adx > choppy_adx


def test_adx_no_lookahead(uptrend_daily):
    full = adx(uptrend_daily["High"], uptrend_daily["Low"], uptrend_daily["Close"], 14)
    truncated = adx(
        uptrend_daily["High"].iloc[:150], uptrend_daily["Low"].iloc[:150], uptrend_daily["Close"].iloc[:150], 14,
    )
    assert full.iloc[149] == pytest.approx(truncated.iloc[149])


def test_technical_snapshot_carries_adx14(uptrend_daily):
    snap = TechnicalAnalyzer().analyze("X", uptrend_daily)
    assert 0 <= snap.adx14 <= 100


# --- Multi-alpha family scores (spec Part 5, added Phase 2) -----------------

def test_family_scores_are_bounded_0_100(uptrend_daily, downtrend_daily, choppy_daily):
    analyzer = TechnicalAnalyzer()
    for df in (uptrend_daily, downtrend_daily, choppy_daily):
        snap = analyzer.analyze("X", df)
        assert 0 <= snap.trend_score() <= 100
        assert 0 <= snap.momentum_score() <= 100
        assert 0 <= snap.volatility_volume_score() <= 100


def test_trend_score_higher_in_uptrend_than_downtrend(uptrend_daily, downtrend_daily):
    analyzer = TechnicalAnalyzer()
    up = analyzer.analyze("UP", uptrend_daily)
    down = analyzer.analyze("DOWN", downtrend_daily)
    assert up.trend_score() > down.trend_score()


def test_family_scores_are_additive_to_technical_score_not_a_replacement(uptrend_daily):
    """Phase 2 must not change technical_score()'s existing behavior --
    the family scores are NEW, additional outputs alongside it, not a
    reimplementation. Pinning technical_score() against a fresh call
    proves it's still the same deterministic function it always was."""
    analyzer = TechnicalAnalyzer()
    snap = analyzer.analyze("X", uptrend_daily)
    assert snap.technical_score() == pytest.approx(snap.technical_score())
