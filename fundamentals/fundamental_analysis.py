"""
Fundamental Analysis module.

Pulls whatever fundamental data is reliably available (yfinance's `info` and
quarterly financial statements by default) and turns it into a bounded,
explainable score. Because free fundamental data is frequently missing,
delayed, or inconsistent across symbols, this module treats data quality as
a first-class citizen: every snapshot records how much data was available
and how stale it is, and the score is discounted (never invented) when
inputs are missing.

"Do not use stale or unreliable fundamental data" is enforced by:
  * capping the contribution of any single missing/old field to 0 (neutral),
    never guessing a value,
  * exposing `data_quality` (0-1) so the signal engine can down-weight the
    fundamentals component entirely when quality is too low,
  * flagging `is_stale` when the last known fiscal update is older than
    `max_age_days`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class FundamentalsUnavailableError(RuntimeError):
    pass


def _default_fetch_fundamentals(symbol: str) -> Dict:
    """Default fetch backed by yfinance. Returns a plain dict so it's easy
    to fake in tests. Any single missing field is tolerated; a totally
    empty response raises so callers can treat the symbol as ungraded."""
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    info = ticker.info or {}
    if not info:
        raise FundamentalsUnavailableError(f"No fundamental info for {symbol}")

    last_update = None
    try:
        ts = info.get("lastFiscalYearEnd") or info.get("mostRecentQuarter")
        if ts:
            last_update = datetime.fromtimestamp(ts, tz=timezone.utc)
    except Exception:
        last_update = None

    return {
        "revenue_growth": info.get("revenueGrowth"),
        "earnings_growth": info.get("earningsGrowth"),
        "eps": info.get("trailingEps"),
        "pe_ratio": info.get("trailingPE"),
        "pb_ratio": info.get("priceToBook"),
        "debt_to_equity": info.get("debtToEquity"),
        "roe": info.get("returnOnEquity"),
        # ROCE (Return on Capital Employed) is a common Indian-equity-research
        # metric but is NOT a standard field in yfinance's `info` payload --
        # left as None (missing) rather than approximated from other fields,
        # per this module's "missing stays missing" rule. Populate it
        # yourself via a fetch_fn that reads a real ROCE figure (e.g. from a
        # screener/vendor that publishes it) if you need it scored.
        "roce": info.get("returnOnCapitalEmployed"),
        "profit_margin": info.get("profitMargins"),
        "operating_cash_flow": info.get("operatingCashflow"),
        "free_cash_flow": info.get("freeCashflow"),
        "last_update": last_update,
        # Spec Part 4 dimension breakdown (VALUATION/PROFITABILITY). Like
        # roce above, peg_ratio/ev_to_ebitda are frequently missing for
        # NSE-listed names via yfinance -- left None rather than guessed,
        # and excluded from the data_quality/data_completeness denominator
        # below for the same reason roce is (see FundamentalAnalyzer.analyze).
        "peg_ratio": info.get("pegRatio"),
        "ev_to_ebitda": info.get("enterpriseToEbitda"),
        # operating_margin/current_ratio are usually populated -- included
        # in the completeness denominator.
        "operating_margin": info.get("operatingMargins"),
        "current_ratio": info.get("currentRatio"),
    }


@dataclass
class FundamentalSnapshot:
    symbol: str
    revenue_growth: Optional[float]
    earnings_growth: Optional[float]
    eps: Optional[float]
    pe_ratio: Optional[float]
    pb_ratio: Optional[float]
    debt_to_equity: Optional[float]
    roe: Optional[float]
    profit_margin: Optional[float]
    operating_cash_flow: Optional[float]
    free_cash_flow: Optional[float]
    last_update: Optional[datetime]
    data_quality: float   # 0-1, fraction of fields that were actually available
    is_stale: bool
    roce: Optional[float] = None   # Return on Capital Employed -- see fetch note above; almost
                                    # always None from the default yfinance-backed fetch, and
                                    # deliberately NOT derived/approximated when missing.
    # Spec Part 4 dimension breakdown -- see _default_fetch_fundamentals's
    # comments for which of these are commonly populated vs. often missing.
    peg_ratio: Optional[float] = None
    ev_to_ebitda: Optional[float] = None
    operating_margin: Optional[float] = None
    current_ratio: Optional[float] = None
    # Sector-relative valuation (e.g. "P/E vs. sector median") is explicitly OUT OF SCOPE here:
    # it requires a point-in-time sector constituent list and peer fundamentals, which this
    # module does not have a reliable free source for. Compute it yourself upstream (e.g. in a
    # custom fetch_fn or a wrapper analyzer) if you have peer data, rather than this module
    # guessing a sector median from an incomplete/unverified peer set.

    def fundamental_score(self) -> float:
        """
        0-100 score built only from available fields. Missing fields
        contribute 0 extra points (neutral), never a guess. If data_quality
        is too low or the data is stale, the caller (signal engine) should
        down-weight this score rather than trusting a partial picture.
        """
        score = 50.0
        contributions = 0

        def add(condition_pts):
            nonlocal score, contributions
            if condition_pts is not None:
                score += condition_pts
                contributions += 1

        add(10 if (self.revenue_growth or 0) > 0.10 else (-5 if self.revenue_growth is not None and self.revenue_growth < 0 else None))
        add(10 if (self.earnings_growth or 0) > 0.10 else (-5 if self.earnings_growth is not None and self.earnings_growth < 0 else None))
        if self.pe_ratio is not None:
            if 0 < self.pe_ratio <= 25:
                add(8)
            elif self.pe_ratio > 60:
                add(-8)
        if self.pb_ratio is not None:
            if 0 < self.pb_ratio <= 4:
                add(4)
            elif self.pb_ratio > 10:
                add(-4)
        if self.debt_to_equity is not None:
            if self.debt_to_equity < 100:
                add(8)
            elif self.debt_to_equity > 200:
                add(-10)
        if self.roe is not None:
            if self.roe > 0.15:
                add(8)
            elif self.roe < 0:
                add(-10)
        if self.profit_margin is not None:
            if self.profit_margin > 0.10:
                add(4)
            elif self.profit_margin < 0:
                add(-8)
        if self.free_cash_flow is not None:
            add(4 if self.free_cash_flow > 0 else -6)

        return float(np.clip(score, 0, 100))

    # ------------------------------------------------------------------
    # Spec Part 4 dimension breakdown. Each is an independently computable
    # 0-100 score over a NAMED, non-overlapping subset of fields -- ADDED
    # alongside fundamental_score() above, which is left completely
    # untouched (same "add, don't touch" contract indicators/technical.py's
    # trend_score()/momentum_score()/volatility_volume_score() established).
    # Baseline 50, bounded additive points, missing fields contribute
    # nothing (never guessed) -- identical style to fundamental_score().
    # ------------------------------------------------------------------

    def valuation_score(self) -> float:
        """Is the stock cheap or expensive on the metrics available."""
        score = 50.0
        if self.pe_ratio is not None:
            if 0 < self.pe_ratio <= 25:
                score += 12
            elif self.pe_ratio > 60:
                score -= 12
        if self.pb_ratio is not None:
            if 0 < self.pb_ratio <= 4:
                score += 8
            elif self.pb_ratio > 10:
                score -= 8
        if self.peg_ratio is not None:
            if 0 < self.peg_ratio <= 1.5:
                score += 8  # PEG <= 1 is a classic "growth at a reasonable price" read
            elif self.peg_ratio > 3:
                score -= 8
        if self.ev_to_ebitda is not None:
            if 0 < self.ev_to_ebitda <= 15:
                score += 6
            elif self.ev_to_ebitda > 30:
                score -= 6
        return float(np.clip(score, 0, 100))

    def profitability_score(self) -> float:
        """How efficiently the business turns capital/sales into profit."""
        score = 50.0
        if self.roe is not None:
            if self.roe > 0.15:
                score += 10
            elif self.roe < 0:
                score -= 12
        if self.roce is not None:  # almost always None from the default fetch -- see field comment
            if self.roce > 0.15:
                score += 8
            elif self.roce < 0:
                score -= 10
        if self.operating_margin is not None:
            if self.operating_margin > 0.15:
                score += 8
            elif self.operating_margin < 0:
                score -= 10
        if self.profit_margin is not None:
            if self.profit_margin > 0.10:
                score += 6
            elif self.profit_margin < 0:
                score -= 10
        return float(np.clip(score, 0, 100))

    def growth_score(self) -> float:
        """Revenue/earnings trajectory -- the two growth fields this
        module actually has (a true multi-year "earnings stability" read
        would need historical statements this fetch doesn't pull)."""
        score = 50.0
        if self.revenue_growth is not None:
            if self.revenue_growth > 0.15:
                score += 12
            elif self.revenue_growth < 0:
                score -= 12
        if self.earnings_growth is not None:
            if self.earnings_growth > 0.15:
                score += 13
            elif self.earnings_growth < 0:
                score -= 13
        return float(np.clip(score, 0, 100))

    def balance_sheet_score(self) -> float:
        """Leverage + liquidity + cash generation."""
        score = 50.0
        if self.debt_to_equity is not None:
            if self.debt_to_equity < 50:
                score += 10
            elif self.debt_to_equity > 150:
                score -= 12
        if self.current_ratio is not None:
            if self.current_ratio >= 1.5:
                score += 8
            elif self.current_ratio < 1.0:
                score -= 10  # short-term obligations may exceed short-term assets
        if self.operating_cash_flow is not None:
            score += 6 if self.operating_cash_flow > 0 else -8
        if self.free_cash_flow is not None:
            score += 6 if self.free_cash_flow > 0 else -8
        return float(np.clip(score, 0, 100))

    def quality_score(self) -> float:
        """Earnings/cash-flow quality signal, built ONLY from what this
        free data source actually provides. Promoter/insider shareholding
        and formal corporate-governance indicators (spec Part 4's
        "QUALITY" examples) are NOT standard yfinance fields and have no
        free generic-vendor source available to this module -- honestly
        omitted rather than guessed, same as the roce/promoter-data gaps
        noted elsewhere in this file. What IS used: whether cash flow
        genuinely backs the reported profits (both operating and free cash
        flow positive is a real, if partial, earnings-quality signal), and
        this snapshot's own data completeness (a thinly-populated snapshot
        is inherently a lower-quality basis for any conclusion)."""
        score = 50.0
        if self.operating_cash_flow is not None and self.free_cash_flow is not None:
            if self.operating_cash_flow > 0 and self.free_cash_flow > 0:
                score += 15
            elif self.operating_cash_flow <= 0 and self.free_cash_flow <= 0:
                score -= 15
        score += float(np.clip((self.data_quality - 0.5) * 30, -15, 15))
        return float(np.clip(score, 0, 100))

    def fundamental_risk_score(self) -> float:
        """0-100, HIGHER = riskier -- a deliberately SEPARATE read from
        fundamental_score(). A stock can look attractive (cheap, growing)
        while still carrying real balance-sheet risk; this exists so that
        distinction isn't lost inside one blended number. Built from
        leverage, cash-flow sign, and earnings-growth sign/magnitude --
        the classic financial-distress-adjacent signals available here."""
        risk = 30.0  # baseline: modest risk assumed even with a clean picture
        if self.debt_to_equity is not None:
            if self.debt_to_equity > 200:
                risk += 25
            elif self.debt_to_equity > 100:
                risk += 12
            elif self.debt_to_equity < 50:
                risk -= 10
        if self.free_cash_flow is not None:
            risk += -10 if self.free_cash_flow > 0 else 20
        if self.current_ratio is not None and self.current_ratio < 1.0:
            risk += 15
        if self.earnings_growth is not None and self.earnings_growth < -0.20:
            risk += 15  # a sharp earnings decline is itself a risk signal, independent of leverage
        return float(np.clip(risk, 0, 100))

    def fundamental_confidence(self) -> float:
        """0-1: how much to trust this snapshot's picture -- distinct from
        `data_quality`/`data_completeness`, which only measure field
        completeness. Driven primarily by completeness, penalized for
        staleness, and penalized further when the five dimension scores
        above disagree widely with each other (a snapshot where valuation
        looks great but balance sheet looks terrible is a genuinely less
        confident/more mixed picture than one where every dimension agrees)."""
        confidence = self.data_quality
        if self.is_stale:
            confidence *= 0.5
        dims = [
            self.valuation_score(), self.profitability_score(), self.growth_score(),
            self.balance_sheet_score(), self.quality_score(),
        ]
        spread = float(np.std(dims))  # 0 = all five dimensions agree exactly
        # A spread of 30+ points (dimensions painting very different
        # pictures) roughly halves confidence; scaled linearly, floored at 0.
        confidence *= float(np.clip(1 - spread / 60.0, 0.0, 1.0))
        return float(np.clip(confidence, 0.0, 1.0))

    @property
    def data_completeness(self) -> float:
        """Alias for `data_quality` under the spec's exact naming (Part 4:
        'fundamental_score, fundamental_risk, fundamental_confidence,
        data_completeness') -- same value, no duplicated storage."""
        return self.data_quality


class FundamentalAnalyzer:
    def __init__(
        self,
        fetch_fn: Optional[Callable[[str], Dict]] = None,
        max_age_days: int = 200,
    ):
        self.fetch_fn = fetch_fn or _default_fetch_fundamentals
        self.max_age_days = max_age_days

    def analyze(self, symbol: str) -> FundamentalSnapshot:
        try:
            raw = self.fetch_fn(symbol)
        except FundamentalsUnavailableError:
            raw = {}
        except Exception as exc:
            # Spec section 22 (fail-safe): a fundamentals fetch can fail for
            # reasons that have nothing to do with "no data exists" --
            # network errors, timeouts, provider outages, unexpected
            # payload shapes. Any of those must degrade to "fundamentals
            # unavailable" (missing stays missing, scored as low
            # data_quality) rather than crashing the caller and never
            # reaching the data-quality/NO-TRADE gate at all.
            logger.warning("Fundamentals fetch failed for %s (treating as unavailable): %s", symbol, exc)
            raw = {}

        fields = [
            "revenue_growth", "earnings_growth", "eps", "pe_ratio", "pb_ratio",
            "debt_to_equity", "roe", "profit_margin", "operating_cash_flow",
            "free_cash_flow", "operating_margin", "current_ratio",
        ]
        # roce/peg_ratio/ev_to_ebitda are intentionally excluded from the
        # data_quality (== data_completeness) denominator: they are rarely
        # populated by the default fetch for NSE-listed names, and counting
        # them would make every snapshot's completeness look artificially
        # low for fields this system doesn't really expect to have.
        available = sum(1 for f in fields if raw.get(f) is not None)
        data_quality = available / len(fields)

        last_update = raw.get("last_update")
        is_stale = True
        if last_update is not None:
            age_days = (datetime.now(timezone.utc) - last_update).days
            is_stale = age_days > self.max_age_days

        return FundamentalSnapshot(
            symbol=symbol,
            revenue_growth=raw.get("revenue_growth"),
            earnings_growth=raw.get("earnings_growth"),
            eps=raw.get("eps"),
            pe_ratio=raw.get("pe_ratio"),
            pb_ratio=raw.get("pb_ratio"),
            debt_to_equity=raw.get("debt_to_equity"),
            roe=raw.get("roe"),
            profit_margin=raw.get("profit_margin"),
            operating_cash_flow=raw.get("operating_cash_flow"),
            free_cash_flow=raw.get("free_cash_flow"),
            last_update=last_update,
            data_quality=data_quality,
            is_stale=is_stale,
            roce=raw.get("roce"),
            peg_ratio=raw.get("peg_ratio"),
            ev_to_ebitda=raw.get("ev_to_ebitda"),
            operating_margin=raw.get("operating_margin"),
            current_ratio=raw.get("current_ratio"),
        )
