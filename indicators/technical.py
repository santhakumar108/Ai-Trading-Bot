"""
Technical Analysis module.

Deliberately keeps the indicator set small and non-redundant:
  * Trend:        SMA(20/50/200), EMA(12/26)
  * Momentum:     RSI(14), MACD(12,26,9)
  * Volatility:   ATR(14), Bollinger Bands(20, 2 std)
  * Volume/price: VWAP (intraday), OBV-style volume trend
  * Structure:    support/resistance (see data.market_data), swing trend

RSI and MACD both measure momentum but on different timescales/formulas and
are cheap to compute, so both are kept; a third or fourth momentum oscillator
(e.g. Stochastic + CCI + Williams %R all together) would be redundant and is
intentionally omitted per the "avoid excessive indicators" instruction.

All functions are pure (DataFrame in, values out) and have no side effects,
which makes them easy to unit test and to reuse inside the backtester.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)  # neutral when undefined (e.g. zero volatility window)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


def bollinger_bands(close: pd.Series, window: int = 20, num_std: float = 2.0):
    mid = sma(close, window)
    std = close.rolling(window=window, min_periods=window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    typical_price = (high + low + close) / 3.0
    cum_vol = volume.cumsum().replace(0, np.nan)
    return (typical_price * volume).cumsum() / cum_vol


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).fillna(0).cumsum()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """
    Average Directional Index (Wilder's formula) -- trend STRENGTH,
    0-100, independent of direction (a strong downtrend and a strong
    uptrend both read high; a choppy/directionless market reads low).
    The one indicator explicitly named in the spec that wasn't already
    here (see module docstring's "avoid excessive indicators" note --
    this fills a genuine gap, RSI/MACD/ADX measure three different things:
    momentum speed, momentum vs. its own average, and trend conviction).
    """
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1,
    ).max(axis=1)

    smoothed_tr = tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    smoothed_plus_dm = plus_dm.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    smoothed_minus_dm = minus_dm.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    tr_safe = smoothed_tr.replace(0, np.nan)
    plus_di = 100 * smoothed_plus_dm / tr_safe
    minus_di = 100 * smoothed_minus_dm / tr_safe
    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    adx_s = dx.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    return adx_s.fillna(0.0)


@dataclass
class TechnicalSnapshot:
    symbol: str
    last_close: float
    sma20: float
    sma50: float
    sma200: Optional[float]
    rsi14: float
    macd_hist: float
    macd_bullish_cross: bool
    atr14: float
    atr_pct_of_price: float
    bb_upper: float
    bb_lower: float
    bb_percent_b: float  # 0 = at lower band, 1 = at upper band
    obv_trend: str        # "RISING", "FALLING", "FLAT"
    trend: str             # "UPTREND", "DOWNTREND", "SIDEWAYS"
    adx14: float = 0.0     # trend STRENGTH (0-100), independent of direction

    def technical_score(self) -> float:
        """
        Combine the indicators above into a single 0-100 technical score.
        This is intentionally simple and explainable rather than a black
        box: each component contributes a bounded amount of points, and the
        components are chosen to avoid double-counting the same information
        (trend, momentum, and volatility/volume each contribute once).
        """
        score = 50.0  # neutral baseline

        # Trend component (+/- 20)
        if self.trend == "UPTREND":
            score += 20
        elif self.trend == "DOWNTREND":
            score -= 20

        # Momentum component via RSI (+/- 15), penalizing extremes (overbought/oversold)
        if 45 <= self.rsi14 <= 65:
            score += 5
        elif self.rsi14 > 70:
            score -= 10  # overbought, chasing risk
        elif self.rsi14 < 30:
            score -= 10  # oversold, but not a green light by itself

        # MACD confirmation (+/- 10)
        if self.macd_bullish_cross and self.macd_hist > 0:
            score += 10
        elif (not self.macd_bullish_cross) and self.macd_hist < 0:
            score -= 10

        # Volume/price confirmation (+/- 10)
        if self.obv_trend == "RISING" and self.trend == "UPTREND":
            score += 10
        elif self.obv_trend == "FALLING" and self.trend == "UPTREND":
            score -= 10  # price up, volume not confirming -> suspect
        elif self.obv_trend == "FALLING" and self.trend == "DOWNTREND":
            score -= 5

        # Bollinger position sanity check (+/- 5): chasing the extreme band is
        # penalized slightly, mean-reversion zone rewarded slightly.
        if self.bb_percent_b > 0.95:
            score -= 5
        elif self.bb_percent_b < 0.05:
            score -= 5

        return float(np.clip(score, 0, 100))

    # ------------------------------------------------------------------
    # Multi-alpha family scores (spec Part 5). Each is an independently
    # computable 0-100 score over a NAMED, non-overlapping subset of the
    # signals above -- NOT a recombination of technical_score() (which is
    # left untouched above for backward compatibility with every existing
    # caller/test). These exist so a consumer (cross-sectional ranking, a
    # future ML feature set) can read "how strong is the TREND read" vs
    # "how strong is the MOMENTUM read" separately, instead of only ever
    # seeing one pre-blended number.
    #
    # Spec Part 6 (correlated-signal control): SMA structure, ADX, and the
    # MACD cross all describe the SAME underlying phenomenon (is there a
    # persistent directional move) and are deliberately kept inside this
    # ONE trend_score() rather than exposed as separate SignalEngine
    # components -- strategy/signal_engine.py still only ever sees the one
    # `technical` vote (via technical_score()), so adding ADX here can
    # never inflate `available_independent_signals` by pretending three
    # trend-family indicators are three independent pieces of evidence.
    # ------------------------------------------------------------------

    def trend_score(self) -> float:
        """SMA/EMA structure + ADX (trend strength) + MACD trend
        confirmation. High = a strong, confirmed directional move; near 50
        = no persistent trend (SIDEWAYS and/or weak ADX)."""
        score = 50.0
        if self.trend == "UPTREND":
            score += 20
        elif self.trend == "DOWNTREND":
            score -= 20

        if self.trend != "SIDEWAYS":
            if self.adx14 >= 25:
                score += 10  # ADX confirms the trend has real conviction
            elif self.adx14 < 15:
                score -= 10  # weak ADX undercuts confidence in the trend read

        if self.macd_bullish_cross and self.macd_hist > 0:
            score += 10
        elif (not self.macd_bullish_cross) and self.macd_hist < 0:
            score -= 10

        return float(np.clip(score, 0, 100))

    def momentum_score(self) -> float:
        """RSI level + MACD histogram SIGN (momentum framing -- is
        momentum currently positive/negative -- distinct from
        trend_score()'s MACD *cross event* framing)."""
        score = 50.0
        if 45 <= self.rsi14 <= 65:
            score += 10
        elif self.rsi14 > 70:
            score -= 15  # overbought
        elif self.rsi14 < 30:
            score -= 15  # oversold -- not a green light by itself

        if self.macd_hist > 0:
            score += 10
        elif self.macd_hist < 0:
            score -= 10

        return float(np.clip(score, 0, 100))

    def volatility_volume_score(self) -> float:
        """OBV/volume confirmation of the current trend + Bollinger-band
        extremity. High = volume genuinely backs the move and price isn't
        stretched to a band extreme; low = divergence or an overextended
        move (spec Part 5.3/5.4, kept as one family since they were
        already combined here before this decomposition)."""
        score = 50.0
        if self.obv_trend == "RISING" and self.trend == "UPTREND":
            score += 15
        elif self.obv_trend == "FALLING" and self.trend == "UPTREND":
            score -= 15  # price up, volume not confirming -> suspect
        elif self.obv_trend == "FALLING" and self.trend == "DOWNTREND":
            score -= 5
        elif self.obv_trend == "RISING" and self.trend == "DOWNTREND":
            score -= 5  # price down but volume accumulating -> divergence

        if self.bb_percent_b > 0.95 or self.bb_percent_b < 0.05:
            score -= 10  # chasing a band extreme either direction

        return float(np.clip(score, 0, 100))


class TechnicalAnalyzer:
    """Computes a TechnicalSnapshot from a daily OHLCV DataFrame."""

    def analyze(self, symbol: str, daily: pd.DataFrame) -> TechnicalSnapshot:
        if len(daily) < 20:
            raise ValueError(f"{symbol}: need at least 20 bars for technical analysis, got {len(daily)}")

        close, high, low, volume = daily["Close"], daily["High"], daily["Low"], daily["Volume"]

        sma20_s = sma(close, 20)
        sma50_s = sma(close, 50) if len(daily) >= 50 else pd.Series([np.nan] * len(daily))
        sma200_s = sma(close, 200) if len(daily) >= 200 else pd.Series([np.nan] * len(daily))
        rsi_s = rsi(close, 14)
        macd_line, signal_line, hist = macd(close)
        atr_s = atr(high, low, close, 14)
        bb_upper, bb_mid, bb_lower = bollinger_bands(close, 20, 2.0)
        obv_s = obv(close, volume)
        adx_s = adx(high, low, close, 14)

        last_close = float(close.iloc[-1])
        sma20 = float(sma20_s.iloc[-1])
        sma50 = float(sma50_s.iloc[-1]) if not np.isnan(sma50_s.iloc[-1]) else sma20
        sma200 = float(sma200_s.iloc[-1]) if not np.isnan(sma200_s.iloc[-1]) else None

        bullish_cross = bool(
            len(macd_line) >= 2
            and macd_line.iloc[-2] <= signal_line.iloc[-2]
            and macd_line.iloc[-1] > signal_line.iloc[-1]
        )

        atr_val = float(atr_s.iloc[-1]) if not np.isnan(atr_s.iloc[-1]) else 0.0
        atr_pct = atr_val / last_close if last_close else 0.0

        bb_up = float(bb_upper.iloc[-1]) if not np.isnan(bb_upper.iloc[-1]) else last_close
        bb_lo = float(bb_lower.iloc[-1]) if not np.isnan(bb_lower.iloc[-1]) else last_close
        band_range = (bb_up - bb_lo) or 1e-9
        percent_b = float((last_close - bb_lo) / band_range)

        obv_recent = obv_s.tail(10)
        if len(obv_recent) >= 2:
            obv_slope = obv_recent.iloc[-1] - obv_recent.iloc[0]
            obv_trend = "RISING" if obv_slope > 0 else ("FALLING" if obv_slope < 0 else "FLAT")
        else:
            obv_trend = "FLAT"

        if last_close > sma20 > sma50:
            trend = "UPTREND"
        elif last_close < sma20 < sma50:
            trend = "DOWNTREND"
        else:
            trend = "SIDEWAYS"

        return TechnicalSnapshot(
            symbol=symbol,
            last_close=last_close,
            sma20=sma20,
            sma50=sma50,
            sma200=sma200,
            rsi14=float(rsi_s.iloc[-1]),
            macd_hist=float(hist.iloc[-1]) if not np.isnan(hist.iloc[-1]) else 0.0,
            macd_bullish_cross=bullish_cross,
            atr14=atr_val,
            atr_pct_of_price=atr_pct,
            bb_upper=bb_up,
            bb_lower=bb_lo,
            bb_percent_b=percent_b,
            obv_trend=obv_trend,
            trend=trend,
            adx14=float(adx_s.iloc[-1]) if not np.isnan(adx_s.iloc[-1]) else 0.0,
        )
