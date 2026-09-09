"""
Shared dashboard computations (spec section 18), used by the CLI dashboard
(cli_dashboard.py), the static HTML snapshot (html_report.py), and the
Streamlit dashboard (dashboard.py) alike, so all three surfaces show the
SAME numbers computed the SAME way rather than three slightly different
ad hoc calculations drifting apart.

Nothing here invents data: a market regime or data-quality status that
can't be determined comes back explicitly as "UNKNOWN"/"N/A", never a
guess.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from config.settings import Config
from data.market_data import DataUnavailableError
from paper_trading.engine import PaperTradingEngine, ScanResult

logger = logging.getLogger(__name__)


def compute_market_regime(engine: PaperTradingEngine, config: Config) -> str:
    """Spec section 18: "Market Regime." Reuses the SAME trailing-only,
    point-in-time-safe classifier the backtester uses
    (`backtesting.backtester.classify_regime`) against the configured
    benchmark index's own recent history -- never a separate, ad hoc
    live-only regime rule. Returns 'UNKNOWN' (never fabricated) if no
    benchmark is configured or its data can't be fetched."""
    from backtesting.backtester import classify_regime

    index_symbol = config.universe.index_symbol
    if not index_symbol:
        return "UNKNOWN (no universe.index_symbol configured)"
    try:
        index_daily = engine.market_data.get_daily(index_symbol)
    except DataUnavailableError as exc:
        logger.warning("Market regime unavailable: could not fetch %s: %s", index_symbol, exc)
        return "UNKNOWN (benchmark data unavailable)"
    if len(index_daily) < 51:
        return "UNKNOWN (insufficient benchmark history)"
    return classify_regime(index_daily, len(index_daily) - 1)


@dataclass
class DataQualitySummary:
    ok: int
    degraded: int
    invalid: int
    unknown: int   # scan produced no result at all for a requested symbol (data totally unavailable)
    total: int
    worst_status: str

    def render_text(self) -> str:
        return (
            f"{self.ok} OK, {self.degraded} DEGRADED, {self.invalid} INVALID, {self.unknown} UNKNOWN "
            f"(of {self.total} symbol(s) requested) -- worst status: {self.worst_status}"
        )


_STATUS_SEVERITY = {"OK": 0, "DEGRADED": 1, "INVALID": 2, "UNKNOWN": 2}


def summarize_data_quality(scan_results: List[ScanResult], symbols_requested: Optional[int] = None) -> DataQualitySummary:
    """Spec section 3: 'every run must display data quality status.'
    Aggregates the per-symbol `DataQualityReport`s already computed by
    `PaperTradingEngine.scan_symbol` -- does not re-derive quality itself."""
    counts = {"OK": 0, "DEGRADED": 0, "INVALID": 0}
    worst = "OK"
    for r in scan_results:
        status = r.data_quality.status if r.data_quality else "UNKNOWN"
        counts[status] = counts.get(status, 0) + 1
        if _STATUS_SEVERITY.get(status, 2) > _STATUS_SEVERITY.get(worst, 0):
            worst = status
    total_requested = symbols_requested if symbols_requested is not None else len(scan_results)
    unknown = max(0, total_requested - len(scan_results))
    if unknown > 0:
        worst = "UNKNOWN" if _STATUS_SEVERITY["UNKNOWN"] >= _STATUS_SEVERITY.get(worst, 0) else worst
    return DataQualitySummary(
        ok=counts.get("OK", 0), degraded=counts.get("DEGRADED", 0), invalid=counts.get("INVALID", 0),
        unknown=unknown, total=total_requested, worst_status=worst,
    )


def top_candidates(scan_results: List[ScanResult], top_n: int = 10) -> List[ScanResult]:
    """Spec section 18: 'Top Candidates.' Sorted by overall confidence,
    descending -- ties broken by symbol for a stable, reproducible order."""
    return sorted(scan_results, key=lambda r: (-r.signal.overall_confidence, r.symbol))[:top_n]


def candidate_row(result: ScanResult) -> Dict[str, object]:
    """One flattened row of every per-candidate field spec section 18 asks
    for: Confidence, Technical/Fundamental/News/Social/ML/Risk Score,
    Entry/Stop/Target/R:R/Expected Value, Decision, Decision Reasons, and
    this candidate's own Data Quality status. `None` fields render as
    'N/A' by the callers below -- never a fabricated number.

    ML score reflects `TradeReport.ml_probability_up` (a 0-1 probability),
    rescaled to the same 0-100 scale as the other score columns here --
    only non-None when the scan that produced this result was run with an
    `ml_provider` supplied (opt-in via --ml on scan/paper-trade/dashboard).
    `None` here means either no --ml flag was used for this run, or the
    model had insufficient history/didn't clear its AUC gate -- never
    fabricated either way.
    """
    rep = result.report
    reasons = rep.reasons[:3]
    return {
        "symbol": result.symbol,
        "price": rep.current_price,
        "decision": rep.decision,
        "confidence": rep.overall_confidence,
        "confidence_label": rep.confidence_label,
        "technical_score": rep.technical_score,
        "fundamental_score": rep.fundamental_score,
        "news_score": rep.news_score,
        "social_score": rep.social_score,
        "ml_score": rep.ml_probability_up * 100 if rep.ml_probability_up is not None else None,
        "risk_score": rep.risk_score,
        "entry": rep.entry,
        "stop_loss": rep.stop_loss,
        "target": rep.target,
        "risk_reward": rep.risk_reward,
        "expected_value_per_share": rep.expected_value_per_share,
        "expected_value_total": rep.expected_value_total,
        "data_quality_status": result.data_quality.status if result.data_quality else "UNKNOWN",
        "reasons": reasons,
    }


def fmt_num(value: Optional[float], spec: str = ".2f") -> str:
    return format(value, spec) if value is not None else "N/A"
