"""
config/nse_all_equity.py: the full NSE EQ-series universe used by the
weekly full-universe scan (scripts/run_weekly_full_scan.ps1,
config/full_universe.yaml). A data-integrity regression guard, not
application logic -- catches an accidental hand-edit introducing a
duplicate or malformed entry.
"""

from config.nse_all_equity import NSE_ALL_EQUITY, SNAPSHOT_DATE


def test_nse_all_equity_has_no_duplicates():
    assert len(NSE_ALL_EQUITY) == len(set(NSE_ALL_EQUITY))


def test_nse_all_equity_every_symbol_is_ns_suffixed():
    assert all(s.endswith(".NS") for s in NSE_ALL_EQUITY)


def test_nse_all_equity_no_blank_or_whitespace_symbols():
    assert all(s == s.strip() and len(s) > len(".NS") for s in NSE_ALL_EQUITY)


def test_nse_all_equity_count_matches_documented_size():
    # Regression guard: catches an accidental partial edit. Update this
    # number (and the module docstring) deliberately if the list is
    # ever intentionally re-sourced/refreshed.
    assert len(NSE_ALL_EQUITY) == 1367


def test_nse_all_equity_snapshot_date_is_documented():
    assert SNAPSHOT_DATE and isinstance(SNAPSHOT_DATE, str)
