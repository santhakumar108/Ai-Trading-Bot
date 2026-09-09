from datetime import datetime, timedelta, timezone

import pytest

from fundamentals.fundamental_analysis import FundamentalAnalyzer, FundamentalsUnavailableError


def fake_fetch_good(symbol):
    return {
        "revenue_growth": 0.18, "earnings_growth": 0.22, "eps": 10.0, "pe_ratio": 18,
        "pb_ratio": 3.0, "debt_to_equity": 40, "roe": 0.25, "profit_margin": 0.20,
        "operating_cash_flow": 1_000_000, "free_cash_flow": 600_000,
        "operating_margin": 0.22, "current_ratio": 1.8,
        "last_update": datetime.now(timezone.utc) - timedelta(days=30),
    }


def fake_fetch_stale(symbol):
    data = fake_fetch_good(symbol)
    data["last_update"] = datetime.now(timezone.utc) - timedelta(days=500)
    return data


def fake_fetch_empty(symbol):
    raise FundamentalsUnavailableError("no data")


def fake_fetch_bad(symbol):
    return {
        "revenue_growth": -0.15, "earnings_growth": -0.25, "eps": -2.0, "pe_ratio": 90,
        "pb_ratio": 15, "debt_to_equity": 300, "roe": -0.2, "profit_margin": -0.1,
        "operating_cash_flow": -500_000, "free_cash_flow": -900_000,
        "operating_margin": -0.05, "current_ratio": 0.6,
        "last_update": datetime.now(timezone.utc) - timedelta(days=10),
    }


def test_good_fundamentals_score_above_neutral():
    analyzer = FundamentalAnalyzer(fetch_fn=fake_fetch_good)
    snap = analyzer.analyze("ACME")
    assert snap.data_quality == 1.0
    assert snap.is_stale is False
    assert snap.fundamental_score() > 50


def test_stale_data_flagged():
    analyzer = FundamentalAnalyzer(fetch_fn=fake_fetch_stale, max_age_days=200)
    snap = analyzer.analyze("ACME")
    assert snap.is_stale is True


def test_missing_data_handled_without_crashing():
    analyzer = FundamentalAnalyzer(fetch_fn=fake_fetch_empty)
    snap = analyzer.analyze("ACME")
    assert snap.data_quality == 0.0
    assert snap.is_stale is True
    assert snap.fundamental_score() == pytest.approx(50.0)  # fully neutral, no fields available


def test_poor_fundamentals_score_below_neutral():
    analyzer = FundamentalAnalyzer(fetch_fn=fake_fetch_bad)
    snap = analyzer.analyze("ACME")
    assert snap.fundamental_score() < 50


# =============================================================================
# Spec Part 4: dimension breakdown -- fundamental_score() regression
# =============================================================================

def test_fundamental_score_unchanged_by_dimension_breakdown_good():
    """Pinned regression: fundamental_score() must be byte-for-byte the
    same as before this phase's dimension methods were added -- raw points
    (50+10+10+8+4+8+8+4+4=106) clip to 100."""
    analyzer = FundamentalAnalyzer(fetch_fn=fake_fetch_good)
    snap = analyzer.analyze("ACME")
    assert snap.fundamental_score() == pytest.approx(100.0)


def test_fundamental_score_unchanged_by_dimension_breakdown_bad():
    """Raw points (50-5-5-8-4-10-10-8-6=-6) clip to 0."""
    analyzer = FundamentalAnalyzer(fetch_fn=fake_fetch_bad)
    snap = analyzer.analyze("ACME")
    assert snap.fundamental_score() == pytest.approx(0.0)


# =============================================================================
# Dimension scores: bounds + directional sensitivity
# =============================================================================

@pytest.mark.parametrize("fetch_fn", [fake_fetch_good, fake_fetch_bad, fake_fetch_empty])
def test_all_dimension_scores_bounded_0_100(fetch_fn):
    analyzer = FundamentalAnalyzer(fetch_fn=fetch_fn)
    snap = analyzer.analyze("ACME")
    for method in (snap.valuation_score, snap.profitability_score, snap.growth_score,
                   snap.balance_sheet_score, snap.quality_score, snap.fundamental_risk_score):
        assert 0 <= method() <= 100


def test_valuation_score_higher_for_cheap_stock():
    good = FundamentalAnalyzer(fetch_fn=fake_fetch_good).analyze("ACME")
    bad = FundamentalAnalyzer(fetch_fn=fake_fetch_bad).analyze("ACME")
    assert good.valuation_score() > bad.valuation_score()


def test_profitability_score_higher_for_profitable_stock():
    good = FundamentalAnalyzer(fetch_fn=fake_fetch_good).analyze("ACME")
    bad = FundamentalAnalyzer(fetch_fn=fake_fetch_bad).analyze("ACME")
    assert good.profitability_score() > bad.profitability_score()


def test_growth_score_higher_for_growing_stock():
    good = FundamentalAnalyzer(fetch_fn=fake_fetch_good).analyze("ACME")
    bad = FundamentalAnalyzer(fetch_fn=fake_fetch_bad).analyze("ACME")
    assert good.growth_score() > bad.growth_score()


def test_balance_sheet_score_higher_for_low_leverage_stock():
    good = FundamentalAnalyzer(fetch_fn=fake_fetch_good).analyze("ACME")
    bad = FundamentalAnalyzer(fetch_fn=fake_fetch_bad).analyze("ACME")
    assert good.balance_sheet_score() > bad.balance_sheet_score()


def test_quality_score_higher_when_cash_flow_backs_profits():
    good = FundamentalAnalyzer(fetch_fn=fake_fetch_good).analyze("ACME")
    bad = FundamentalAnalyzer(fetch_fn=fake_fetch_bad).analyze("ACME")
    assert good.quality_score() > bad.quality_score()


def test_quality_score_never_uses_promoter_or_governance_data():
    """Spec Part 4 lists promoter/governance signals -- this module has no
    free source for them; confirms quality_score() doesn't silently depend
    on any such field (there are none on the dataclass to depend on)."""
    import dataclasses
    from fundamentals.fundamental_analysis import FundamentalSnapshot
    field_names = {f.name for f in dataclasses.fields(FundamentalSnapshot)}
    assert not any("promoter" in f.lower() or "governance" in f.lower() or "shareholding" in f.lower()
                   for f in field_names)


# =============================================================================
# fundamental_risk_score
# =============================================================================

def test_fundamental_risk_score_higher_for_leveraged_stock():
    good = FundamentalAnalyzer(fetch_fn=fake_fetch_good).analyze("ACME")
    bad = FundamentalAnalyzer(fetch_fn=fake_fetch_bad).analyze("ACME")
    assert bad.fundamental_risk_score() > good.fundamental_risk_score()


def test_fundamental_risk_score_is_a_separate_concept_from_fundamental_score():
    """A stock can score well on fundamental_score() while still being
    flagged higher-risk -- confirms the two aren't just mirror images by
    checking they don't always move in lockstep across both fixtures."""
    good = FundamentalAnalyzer(fetch_fn=fake_fetch_good).analyze("ACME")
    bad = FundamentalAnalyzer(fetch_fn=fake_fetch_bad).analyze("ACME")
    # Good fundamentals -> higher fundamental_score AND lower risk (both
    # move the "right" way here), but computed via genuinely different
    # inputs/formulas -- not simply `100 - fundamental_score()`.
    assert good.fundamental_score() != pytest.approx(100 - good.fundamental_risk_score())


def test_fundamental_risk_score_high_on_sharp_earnings_decline():
    def fetch(symbol):
        return {"earnings_growth": -0.30, "last_update": datetime.now(timezone.utc)}
    snap = FundamentalAnalyzer(fetch_fn=fetch).analyze("ACME")
    baseline = FundamentalAnalyzer(fetch_fn=lambda s: {"last_update": datetime.now(timezone.utc)}).analyze("ACME")
    assert snap.fundamental_risk_score() > baseline.fundamental_risk_score()


# =============================================================================
# fundamental_confidence
# =============================================================================

def test_confidence_lower_when_stale():
    fresh = FundamentalAnalyzer(fetch_fn=fake_fetch_good, max_age_days=200).analyze("ACME")
    stale = FundamentalAnalyzer(fetch_fn=fake_fetch_stale, max_age_days=200).analyze("ACME")
    assert stale.fundamental_confidence() < fresh.fundamental_confidence()


def test_confidence_lower_when_data_missing():
    complete = FundamentalAnalyzer(fetch_fn=fake_fetch_good).analyze("ACME")
    empty = FundamentalAnalyzer(fetch_fn=fake_fetch_empty).analyze("ACME")
    assert empty.fundamental_confidence() < complete.fundamental_confidence()


def test_confidence_bounded_0_1():
    for fetch_fn in (fake_fetch_good, fake_fetch_bad, fake_fetch_empty, fake_fetch_stale):
        snap = FundamentalAnalyzer(fetch_fn=fetch_fn).analyze("ACME")
        assert 0.0 <= snap.fundamental_confidence() <= 1.0


def test_confidence_lower_when_dimensions_disagree_widely():
    """A snapshot where valuation looks great but the balance sheet looks
    terrible is a genuinely more mixed/less confident picture than one
    where every available dimension agrees."""
    def fetch_mixed(symbol):
        return {
            # Extremely cheap valuation...
            "pe_ratio": 5, "pb_ratio": 0.5, "peg_ratio": 0.3,
            # ...but alarming balance sheet.
            "debt_to_equity": 400, "current_ratio": 0.3, "free_cash_flow": -2_000_000,
            "last_update": datetime.now(timezone.utc),
        }

    def fetch_consistent(symbol):
        return {
            "pe_ratio": 18, "pb_ratio": 3.0, "roe": 0.18, "profit_margin": 0.12,
            "debt_to_equity": 60, "current_ratio": 1.6, "free_cash_flow": 500_000,
            "operating_cash_flow": 700_000, "revenue_growth": 0.10, "earnings_growth": 0.10,
            "last_update": datetime.now(timezone.utc),
        }

    mixed = FundamentalAnalyzer(fetch_fn=fetch_mixed).analyze("ACME")
    consistent = FundamentalAnalyzer(fetch_fn=fetch_consistent).analyze("ACME")
    assert mixed.fundamental_confidence() < consistent.fundamental_confidence()


# =============================================================================
# data_completeness alias
# =============================================================================

@pytest.mark.parametrize("fetch_fn", [fake_fetch_good, fake_fetch_bad, fake_fetch_empty])
def test_data_completeness_always_equals_data_quality(fetch_fn):
    snap = FundamentalAnalyzer(fetch_fn=fetch_fn).analyze("ACME")
    assert snap.data_completeness == snap.data_quality


def test_data_completeness_excludes_roce_peg_ev_ebitda():
    """These three fields are rarely populated by the default fetch --
    their absence must NOT drag completeness down (same treatment)."""
    def fetch(symbol):
        d = fake_fetch_good(symbol)
        # roce/peg_ratio/ev_to_ebitda deliberately absent.
        return d
    snap = FundamentalAnalyzer(fetch_fn=fetch).analyze("ACME")
    assert snap.data_completeness == 1.0
    assert snap.roce is None and snap.peg_ratio is None and snap.ev_to_ebitda is None
