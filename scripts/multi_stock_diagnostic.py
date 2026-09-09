"""
Multi-stock zero-trade diagnostic (spec sections 1-7 & 11-12 of the "IMPORTANT
NEXT FIX" request).

IMPORTANT HONESTY NOTE -- read before trusting these numbers:
This sandboxed environment cannot reach Yahoo Finance / any real NSE data
vendor (outbound requests to query1.finance.yahoo.com are blocked at the
network layer -- confirmed directly), and no linked device with independent
network access was available either (`get_device_info` reported zero
connected folders). This script therefore uses DETERMINISTIC, CLEARLY
SYNTHETIC 5-year daily OHLCV series -- one per diagnostic symbol, seeded and
shaped differently (strong uptrend / mild uptrend / downtrend / choppy /
high-volatility) to stand in for "different sectors and liquidity levels" --
NOT real historical prices. This satisfies the STRUCTURAL half of the
request (is the pipeline itself broken for every symbol identically,
regardless of what the price series looks like?) but does NOT satisfy the
literal "verify real historical OHLCV data" ask in section 8 end-to-end
against a live vendor. See the final report for what this does and does not
prove, and README.md's "Known issue, fixed" section for the same caveat.

The 10 symbols below (TCS.NS, INFY.NS, ...) are used ONLY as diagnostic
labels attached to synthetic series -- this is explicitly NOT a claim that
this is real TCS/INFY/... price history, and these are NOT hard-coded
anywhere as the system's permanent trading universe (see
config/default_config.yaml's `universe.symbols`, which is unchanged).

Usage:
    python scripts/multi_stock_diagnostic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtesting.diagnostics import SignalFunnel, render_summary_table
from config.settings import Config
from tests.conftest import make_synthetic_ohlcv

# (symbol, seed, drift, volatility) -- deliberately varied, not all bullish.
# Chosen to represent different "sectors and liquidity levels" as requested,
# using different (fake) drift/volatility regimes rather than real sector
# betas, which this environment has no way to source honestly.
DIAGNOSTIC_UNIVERSE = [
    ("TCS.NS",        1001, 0.0035, 0.011),   # strong, low-noise uptrend
    ("INFY.NS",       1002, 0.0020, 0.014),   # mild uptrend
    ("RELIANCE.NS",   1003, 0.0006, 0.016),   # weak drift, moderate noise
    ("HDFCBANK.NS",   1004, 0.0028, 0.012),   # uptrend, bank-like lower vol
    ("ICICIBANK.NS",  1005, 0.0000, 0.018),   # choppy / sideways
    ("SBIN.NS",       1006, -0.0010, 0.020),  # mild downtrend
    ("ITC.NS",        1007, 0.0004, 0.009),   # low-vol, low-drift (FMCG-like)
    ("LT.NS",         1008, 0.0018, 0.017),   # uptrend, higher vol (capital goods-like)
    ("AXISBANK.NS",   1009, -0.0025, 0.022),  # downtrend, higher vol
    ("BHARTIARTL.NS", 1010, 0.0040, 0.013),   # strong uptrend, telecom-like
]

N_BARS = 1250  # ~5 years of NSE trading days (~250/yr)


def build_universe():
    frames = {}
    for symbol, seed, drift, vol in DIAGNOSTIC_UNIVERSE:
        frames[symbol] = make_synthetic_ohlcv(n=N_BARS, drift=drift, volatility=vol, seed=seed, start_date="2021-01-04")
    index_daily = make_synthetic_ohlcv(n=N_BARS, drift=0.0008, volatility=0.012, seed=999, start_date="2021-01-04")
    return frames, index_daily


def run(config: Config, label: str):
    frames, index_daily = build_universe()
    reports = []
    for symbol, *_ in DIAGNOSTIC_UNIVERSE:
        funnel = SignalFunnel(config=config)
        report = funnel.run(symbol, frames[symbol], index_daily=index_daily)
        reports.append(report)

    print(f"\n{'=' * 100}\n{label}  (min_independent_signals={config.decision_thresholds.min_independent_signals})\n{'=' * 100}")
    print(render_summary_table(reports))
    for r in reports:
        c = r.counts
        print(f"\n--- {r.symbol} full stage funnel ---")
        print(f"  Data quality: {r.data_quality.status} (score={r.data_quality.quality_score:.2f}, tradeable={r.data_quality_tradeable})")
        print(f"  Bars total:                         {c.bars_total}")
        print(f"  Indicator warm-up ok (>=20 bars):    {c.bars_with_min_bars_for_indicators}  (full SMA200 warm-up: {c.indicator_warmup_complete})")
        print(f"  Technical: bullish/bearish/neutral:  {c.technical_bullish} / {c.technical_bearish} / {c.technical_neutral}")
        print(f"  Regime: BULL/BEAR/SIDEWAYS/UNKNOWN:  {c.regime_bull} / {c.regime_bear} / {c.regime_sideways} / {c.regime_unknown}  (informational only -- never a gate)")
        print(f"  NO TRADE -- insufficient history:    {c.no_trade_insufficient_history}")
        print(f"  NO TRADE -- no components available: {c.no_trade_no_components_available}")
        print(f"  NO TRADE -- independent-signals gate:{c.no_trade_independent_signals_gate}")
        print(f"  NO TRADE -- disagreement gate:       {c.no_trade_disagreement_gate}")
        print(f"  Confidence pass/fail:                {c.confidence_pass} / {c.confidence_fail}")
        print(f"  Signal aligned (BUY/SELL from engine):{c.signal_aligned}")
        print(f"  Trade setup computed (risk assessed): {c.trade_setup_computed}")
        print(f"  R:R pass/fail:                       {c.risk_reward_pass} / {c.risk_reward_fail}")
        print(f"  Expected value pass/fail:             {c.expected_value_pass} / {c.expected_value_fail}")
        print(f"  Account risk limits pass/fail:        {c.account_risk_limits_pass} / {c.account_risk_limits_fail}")
        print(f"  FINAL APPROVED:                       {c.final_approved}")
        if r.example_approved_bar:
            print(f"  Example approved bar: {r.example_approved_bar}")
        elif r.rejection_reason_samples:
            print(f"  Sample rejection reasons: {r.rejection_reason_samples[:3]}")
    return reports


if __name__ == "__main__":
    before_config = Config()
    before_config.decision_thresholds.min_independent_signals = 2  # the OLD (buggy) default, for comparison
    run(before_config, "BEFORE FIX")

    after_config = Config()  # min_independent_signals now defaults to 1 -- see config/settings.py
    run(after_config, "AFTER FIX")
