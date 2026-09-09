"""
Static NSE universe presets: NIFTY 50, a best-effort NIFTY Next 50, and a
sector -> index-symbol map.

Honesty note (read before using these in anything beyond a starting point):
Index constituents are NOT fetched live in this system -- there is no
free, reliable, machine-readable "current NIFTY 50 constituents" feed this
codebase pulls from, and NSE reconstitutes these indices periodically
(typically semi-annually). The lists below are a hand-maintained SNAPSHOT
(see `SNAPSHOT_DATE`), good enough to seed `universe.symbols` for research,
but they WILL drift out of date. For anything you rely on operationally:
  * Refresh this list yourself from NSE's official published index
    constituent files (nseindia.com publishes these; this environment has
    no network path to fetch them automatically), and/or
  * Point `universe.symbols` at a list you maintain and update on your own
    schedule -- this module is a convenience default, not a live feed.

Yahoo Finance / yfinance ticker convention for NSE-listed equities is the
company's NSE trading symbol plus ".NS" (e.g. "RELIANCE.NS"); BSE listings
use ".BO". Index tickers are yfinance's own symbols, not NSE's own quote
codes (e.g. NIFTY 50 is "^NSEI", not "NIFTY50").
"""

from __future__ import annotations

from typing import Dict, List

SNAPSHOT_DATE = "2024-Q2"  # when this list was last hand-checked; refresh periodically

# NIFTY 50 constituents as of the snapshot date above, in NSE trading-symbol
# form with the ".NS" suffix yfinance expects. Order is not significant.
NIFTY_50: List[str] = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "BAJFINANCE.NS",
    "KOTAKBANK.NS", "LT.NS", "HCLTECH.NS", "AXISBANK.NS", "ASIANPAINT.NS",
    "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "BAJAJFINSV.NS",
    "WIPRO.NS", "NESTLEIND.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS",
    "M&M.NS", "TATASTEEL.NS", "TATAMOTORS.NS", "ADANIENT.NS", "COALINDIA.NS",
    "JSWSTEEL.NS", "INDUSINDBK.NS", "GRASIM.NS", "TECHM.NS", "HINDALCO.NS",
    "CIPLA.NS", "DRREDDY.NS", "BRITANNIA.NS", "EICHERMOT.NS", "APOLLOHOSP.NS",
    "DIVISLAB.NS", "BAJAJ-AUTO.NS", "SBILIFE.NS", "HDFCLIFE.NS", "BPCL.NS",
    "HEROMOTOCO.NS", "UPL.NS", "TATACONSUM.NS", "SHRIRAMFIN.NS", "LTIM.NS",
]

# NIFTY Next 50: lower confidence than NIFTY_50 above -- this tier turns
# over more often. Treat as an even rougher starting point.
NIFTY_NEXT_50: List[str] = [
    "ADANIGREEN.NS", "ADANIPOWER.NS", "AMBUJACEM.NS", "BANKBARODA.NS",
    "BERGEPAINT.NS", "BOSCHLTD.NS", "CANBK.NS", "CHOLAFIN.NS", "COLPAL.NS",
    "DABUR.NS", "DLF.NS", "GAIL.NS", "GODREJCP.NS", "HAVELLS.NS",
    "ICICIGI.NS", "ICICIPRULI.NS", "IOC.NS", "IRCTC.NS", "JINDALSTEL.NS",
    "LICI.NS", "MARICO.NS", "MOTHERSON.NS", "NAUKRI.NS", "PIDILITIND.NS",
    "PNB.NS", "SIEMENS.NS", "TATAPOWER.NS", "TORNTPHARM.NS", "TRENT.NS",
    "VEDL.NS", "ZOMATO.NS",
]

# Sector -> NSE sector-index yfinance ticker. Used for `universe.sector_indices`
# so a symbol mapped to a sector (via `universe.sector_map`) gets a real
# sector-trend read instead of "N/A". Verify these still resolve on your
# data vendor before relying on them -- Yahoo occasionally renames/retires
# index tickers.
NSE_SECTOR_INDICES: Dict[str, str] = {
    "IT": "^CNXIT",
    "BANK": "^NSEBANK",
    "AUTO": "^CNXAUTO",
    "PHARMA": "^CNXPHARMA",
    "FMCG": "^CNXFMCG",
    "METAL": "^CNXMETAL",
    "ENERGY": "^CNXENERGY",
    "REALTY": "^CNXREALTY",
    "PSU_BANK": "^CNXPSUBANK",
    "FIN_SERVICE": "^CNXFINANCE",
}

# Symbol -> sector-label map for NIFTY_50 (spec Part 22: cross-stock
# robustness reporting groups by sector -- this is the ONLY thing that
# lets it do so; NSE_SECTOR_INDICES above maps a sector NAME to a live
# benchmark ticker, but nothing previously mapped a SYMBOL to a sector
# name at all). Same honesty caveat as NIFTY_50 above: hand-maintained,
# best-effort, not fetched live, and some conglomerates/diversified names
# are a judgment call (e.g. RELIANCE is core O&G/refining-weighted here
# despite retail/telecom arms; ADANIENT is genuinely multi-business and
# labeled DIVERSIFIED rather than forced into one bucket). Sector labels
# here are NOT required to have a matching NSE_SECTOR_INDICES entry --
# this map is for grouping in aggregate reporting, not for live sector-
# trend scoring (see config/settings.py's UniverseConfig.sector_map for
# that, which stays empty by default and is a separate, deliberate
# per-deployment choice).
NIFTY_50_SECTOR_MAP: Dict[str, str] = {
    "RELIANCE.NS": "ENERGY", "TCS.NS": "IT", "HDFCBANK.NS": "BANK", "ICICIBANK.NS": "BANK",
    "INFY.NS": "IT", "HINDUNILVR.NS": "FMCG", "ITC.NS": "FMCG", "SBIN.NS": "BANK",
    "BHARTIARTL.NS": "TELECOM", "BAJFINANCE.NS": "FIN_SERVICE", "KOTAKBANK.NS": "BANK",
    "LT.NS": "INFRA", "HCLTECH.NS": "IT", "AXISBANK.NS": "BANK", "ASIANPAINT.NS": "CONSUMER",
    "MARUTI.NS": "AUTO", "SUNPHARMA.NS": "PHARMA", "TITAN.NS": "CONSUMER", "ULTRACEMCO.NS": "CEMENT",
    "BAJAJFINSV.NS": "FIN_SERVICE", "WIPRO.NS": "IT", "NESTLEIND.NS": "FMCG", "ONGC.NS": "ENERGY",
    "NTPC.NS": "ENERGY", "POWERGRID.NS": "ENERGY", "M&M.NS": "AUTO", "TATASTEEL.NS": "METAL",
    "TATAMOTORS.NS": "AUTO", "ADANIENT.NS": "DIVERSIFIED", "COALINDIA.NS": "ENERGY",
    "JSWSTEEL.NS": "METAL", "INDUSINDBK.NS": "BANK", "GRASIM.NS": "CEMENT", "TECHM.NS": "IT",
    "HINDALCO.NS": "METAL", "CIPLA.NS": "PHARMA", "DRREDDY.NS": "PHARMA", "BRITANNIA.NS": "FMCG",
    "EICHERMOT.NS": "AUTO", "APOLLOHOSP.NS": "HEALTHCARE", "DIVISLAB.NS": "PHARMA",
    "BAJAJ-AUTO.NS": "AUTO", "SBILIFE.NS": "INSURANCE", "HDFCLIFE.NS": "INSURANCE", "BPCL.NS": "ENERGY",
    "HEROMOTOCO.NS": "AUTO", "UPL.NS": "CHEMICALS", "TATACONSUM.NS": "FMCG",
    "SHRIRAMFIN.NS": "FIN_SERVICE", "LTIM.NS": "IT",
}
