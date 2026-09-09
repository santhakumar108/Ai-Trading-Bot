from datetime import datetime, timedelta, timezone

import pytest

from news.news_analysis import (
    NewsAnalyzer, NewsItem, NewsSourceAdapter, classify_event_type, classify_sentiment_taxonomy,
)


class FakeSource(NewsSourceAdapter):
    def __init__(self, items):
        self._items = items

    def fetch(self, symbol, company_name=None):
        return self._items


def fake_sentiment(text: str) -> float:
    # Deterministic stand-in for the lexicon model: positive if "good" in text.
    if "great" in text.lower():
        return 0.8
    if "terrible" in text.lower():
        return -0.8
    return 0.0


def make_item(headline, domain, hours_ago, symbol="ACME"):
    return NewsItem(
        symbol=symbol, headline=headline, source_domain=domain,
        published_at=datetime.now(timezone.utc) - timedelta(hours=hours_ago),
    )


def test_single_headline_confidence_is_capped():
    items = [make_item("Great earnings beat", "reuters.com", 1)]
    analyzer = NewsAnalyzer(source=FakeSource(items), sentiment_fn=fake_sentiment, min_source_credibility=0.4)
    agg = analyzer.analyze("ACME")
    assert agg.num_credible_sources == 1
    assert agg.overall_confidence <= 35.0, "A single-source headline must not reach high confidence."


def test_contradictory_reports_detected_and_dampened():
    items = [
        make_item("Great new product launch", "reuters.com", 2),
        make_item("Terrible fraud allegations surface", "bloomberg.com", 3),
    ]
    analyzer = NewsAnalyzer(source=FakeSource(items), sentiment_fn=fake_sentiment)
    agg = analyzer.analyze("ACME")
    assert agg.contradictory is True
    assert agg.news_score() == pytest.approx(50.0, abs=15.0)  # pulled toward neutral


def test_multiple_agreeing_credible_sources_raise_confidence():
    items = [
        make_item("Great earnings beat", "reuters.com", 1),
        make_item("Great earnings beat expectations", "bloomberg.com", 2),
        make_item("Great quarter for the company", "wsj.com", 3),
    ]
    analyzer = NewsAnalyzer(source=FakeSource(items), sentiment_fn=fake_sentiment)
    agg = analyzer.analyze("ACME")
    assert agg.num_credible_sources == 3
    assert agg.overall_confidence > 35.0
    assert agg.news_score() > 50.0


def test_stale_news_excluded():
    items = [make_item("Great earnings beat", "reuters.com", hours_ago=200)]  # older than default max age
    analyzer = NewsAnalyzer(source=FakeSource(items), sentiment_fn=fake_sentiment, max_headline_age_hours=48)
    agg = analyzer.analyze("ACME")
    assert agg.num_credible_sources == 0
    assert agg.overall_confidence == 0.0


def test_low_credibility_source_excluded():
    items = [make_item("Great earnings beat", "reddit.com", 1)]
    analyzer = NewsAnalyzer(source=FakeSource(items), sentiment_fn=fake_sentiment, min_source_credibility=0.4)
    agg = analyzer.analyze("ACME")
    assert agg.num_credible_sources == 0


def test_no_news_returns_neutral_zero_confidence():
    analyzer = NewsAnalyzer(source=FakeSource([]), sentiment_fn=fake_sentiment)
    agg = analyzer.analyze("ACME")
    assert agg.overall_confidence == 0.0
    assert agg.news_score() == pytest.approx(50.0)


# --- Relevance, novelty, and duplicate/syndicated-article handling --------

def test_relevant_headline_scores_higher_confidence_than_ambiguous_match():
    relevant = make_item("Acme Corp reports great earnings beat", "reuters.com", 1)
    ambiguous = make_item("Great weather forecast for the weekend", "reuters.com", 1)  # doesn't mention the company
    analyzer = NewsAnalyzer(source=FakeSource([relevant]), sentiment_fn=fake_sentiment, min_source_credibility=0.4)
    analyzer.score_item(relevant, symbol="ACME", company_name="Acme Corp")
    analyzer.score_item(ambiguous, symbol="ACME", company_name="Acme Corp")
    assert relevant.relevance == 1.0
    assert ambiguous.relevance < 1.0
    assert relevant.confidence > ambiguous.confidence


def test_near_duplicate_syndicated_articles_are_deduped_not_double_counted():
    """Two outlets running the SAME wire story (near-identical headline)
    must not count as two independent corroborating sources -- a single
    underlying event should never look like broader confirmation than it is."""
    items = [
        make_item("Company X reports record quarterly profit surge", "reuters.com", 2),
        make_item("Company X reports record quarterly profit surge", "moneycontrol.com", 1),  # wire copy
    ]
    analyzer = NewsAnalyzer(source=FakeSource(items), sentiment_fn=fake_sentiment)
    agg = analyzer.analyze("ACME")
    assert agg.num_credible_sources == 1
    dup_flags = [i.is_duplicate for i in agg.items]
    assert dup_flags.count(True) == 1
    assert dup_flags.count(False) == 1


def test_earliest_copy_of_a_story_is_kept_not_the_duplicate():
    earlier = make_item("Firm announces major acquisition deal", "reuters.com", 5)
    later_copy = make_item("Firm announces major acquisition deal", "bloomberg.com", 1)
    analyzer = NewsAnalyzer(source=FakeSource([earlier, later_copy]), sentiment_fn=fake_sentiment)
    agg = analyzer.analyze("ACME")
    kept = [i for i in agg.items if not i.is_duplicate]
    assert len(kept) == 1
    assert kept[0].source_domain == "reuters.com"  # the earlier one, by publish time


def test_genuinely_distinct_articles_are_not_treated_as_duplicates():
    items = [
        make_item("Company X reports great quarterly profit surge", "reuters.com", 2),
        make_item("Company X terrible fraud allegations surface", "bloomberg.com", 1),
    ]
    analyzer = NewsAnalyzer(source=FakeSource(items), sentiment_fn=fake_sentiment)
    agg = analyzer.analyze("ACME")
    assert agg.contradictory is True
    assert all(not i.is_duplicate for i in agg.items)


# =============================================================================
# Spec Part 3: event classification
# =============================================================================

_EVENT_HEADLINE_CASES = [
    ("Company flags accounting irregularities amid fraud probe", "FRAUD_ALLEGATION"),
    ("SEBI initiates regulatory action against the firm", "REGULATORY_ACTION"),
    ("Firm sued for breach of contract in high court litigation", "LEGAL_ISSUE"),
    ("CEO steps down; board appoints new CFO", "MANAGEMENT_CHANGE"),
    ("Company loses contract with major client, order cancelled", "ORDER_LOSS"),
    ("Firm wins order worth Rs 500 crore from government", "ORDER_WIN"),
    ("Company to acquire smaller rival in all-cash deal", "ACQUISITION"),
    ("Two firms announce merger to create market leader", "MERGER"),
    ("Rating agency downgrades rating on rising debt concerns", "DEBT"),
    ("Promoter pledge of shares rises amid cash crunch", "PROMOTER_ACTIVITY"),
    ("Large block deal reported in company shares today", "INSIDER_ACTIVITY"),
    ("Board declares interim dividend of Rs 5 per share", "DIVIDEND"),
    ("Company announces share buyback worth Rs 1000 crore", "BUYBACK"),
    ("Board approves 1:2 stock split for the company", "SPLIT"),
    ("Company announces bonus issue in ratio 1:1", "BONUS"),
    ("Company announces rights issue to raise capital", "CORPORATE_ACTION"),
    ("Company reports Q3 results, net profit rises sharply", "EARNINGS"),
    ("Management raises outlook for the fiscal year", "GUIDANCE"),
    ("RBI hikes repo rate amid rising inflation concerns", "MACRO_IMPACT"),
    ("Trade war tensions escalate amid new tariff threats", "GEOPOLITICAL_EVENT"),
    ("Industry-wide slowdown hits sector margins", "SECTOR_EVENT"),
    ("Local cricket team wins regional tournament", "OTHER"),
]


@pytest.mark.parametrize("headline,expected", _EVENT_HEADLINE_CASES)
def test_classify_event_type_matches_expected_category(headline, expected):
    assert classify_event_type(headline) == expected


def test_classify_event_type_wins_bid_is_order_win():
    """Caught via a live-data check against real headlines: 'wins bid'
    phrasing (as opposed to 'wins order'/'wins contract') was initially
    missed -- a real, honest keyword-coverage gap, not a synthetic case."""
    assert classify_event_type("TCS Wins INR1.23 Billion Odisha State Workflow Automation Bid") == "ORDER_WIN"


def test_classify_event_type_uses_summary_too():
    assert classify_event_type("Company update", "Firm announces major acquisition of a rival") == "ACQUISITION"


def test_classify_event_type_unmatched_headline_is_other():
    assert classify_event_type("A perfectly ordinary sentence about nothing in particular") == "OTHER"


def test_fraud_takes_priority_over_generic_earnings_wording():
    """A headline that could loosely resemble more than one category
    resolves to the more SPECIFIC one -- fraud allegations, even when
    phrased near financial-results language, must not be swallowed by the
    generic EARNINGS bucket."""
    headline = "Fraud allegations surface just before quarterly results announcement"
    assert classify_event_type(headline) == "FRAUD_ALLEGATION"


def test_score_item_populates_event_type_and_timeframe():
    analyzer = NewsAnalyzer(sentiment_fn=fake_sentiment)
    item = make_item("Board declares interim dividend of Rs 5 per share", "reuters.com", 1)
    scored = analyzer.score_item(item, symbol="ACME")
    assert scored.event_type == "DIVIDEND"
    assert scored.affected_timeframe == "IMMEDIATE_CORPORATE_ACTION"


def test_score_item_earnings_maps_to_quarter_timeframe():
    analyzer = NewsAnalyzer(sentiment_fn=fake_sentiment)
    item = make_item("Company reports Q3 results, net profit rises sharply", "reuters.com", 1)
    scored = analyzer.score_item(item, symbol="ACME")
    assert scored.event_type == "EARNINGS"
    assert scored.affected_timeframe == "QUARTER"


def test_score_item_other_maps_to_short_term_timeframe():
    analyzer = NewsAnalyzer(sentiment_fn=fake_sentiment)
    item = make_item("Local cricket team wins regional tournament", "reuters.com", 1)
    scored = analyzer.score_item(item, symbol="ACME")
    assert scored.event_type == "OTHER"
    assert scored.affected_timeframe == "SHORT_TERM"


# =============================================================================
# Spec Part 3: sentiment taxonomy (item level)
# =============================================================================

def test_classify_sentiment_taxonomy_thresholds():
    assert classify_sentiment_taxonomy(0.5) == "POSITIVE"
    assert classify_sentiment_taxonomy(0.2) == "POSITIVE"
    assert classify_sentiment_taxonomy(0.1) == "NEUTRAL"
    assert classify_sentiment_taxonomy(0.0) == "NEUTRAL"
    assert classify_sentiment_taxonomy(-0.1) == "NEUTRAL"
    assert classify_sentiment_taxonomy(-0.2) == "NEGATIVE"
    assert classify_sentiment_taxonomy(-0.5) == "NEGATIVE"


def test_item_sentiment_class_positive_negative_neutral():
    analyzer = NewsAnalyzer(sentiment_fn=fake_sentiment)
    positive = analyzer.score_item(make_item("Great news for the company", "reuters.com", 1), symbol="ACME")
    negative = analyzer.score_item(make_item("Terrible news for the company", "reuters.com", 1), symbol="ACME")
    neutral = analyzer.score_item(make_item("Company issues routine update", "reuters.com", 1), symbol="ACME")
    assert positive.sentiment_class == "POSITIVE"
    assert negative.sentiment_class == "NEGATIVE"
    assert neutral.sentiment_class == "NEUTRAL"


def test_item_sentiment_class_unknown_when_no_text():
    analyzer = NewsAnalyzer(sentiment_fn=fake_sentiment)
    empty_item = NewsItem(
        symbol="ACME", headline="", source_domain="reuters.com",
        published_at=datetime.now(timezone.utc), summary="",
    )
    scored = analyzer.score_item(empty_item, symbol="ACME")
    assert scored.sentiment_class == "UNKNOWN"


# =============================================================================
# Spec Part 3: sentiment taxonomy (aggregate level) -- MIXED/UNKNOWN
# =============================================================================

def test_aggregate_sentiment_class_mixed_tracks_existing_contradictory_flag():
    items = [
        make_item("Company X reports great quarterly profit surge", "reuters.com", 2),
        make_item("Company X terrible fraud allegations surface", "bloomberg.com", 1),
    ]
    analyzer = NewsAnalyzer(source=FakeSource(items), sentiment_fn=fake_sentiment)
    agg = analyzer.analyze("ACME")
    assert agg.contradictory is True
    assert agg.sentiment_class == "MIXED"


def test_aggregate_sentiment_class_unknown_with_no_credible_sources():
    items = [make_item("Great earnings", "reddit.com", 1)]  # below default credibility floor
    analyzer = NewsAnalyzer(source=FakeSource(items), sentiment_fn=fake_sentiment, min_source_credibility=0.9)
    agg = analyzer.analyze("ACME")
    assert agg.num_credible_sources == 0
    assert agg.sentiment_class == "UNKNOWN"


def test_aggregate_sentiment_class_positive_when_agreeing_and_positive():
    items = [
        make_item("Great earnings beat for the company", "reuters.com", 1),
        make_item("Great earnings beat confirmed by another outlet entirely", "bloomberg.com", 2),
    ]
    analyzer = NewsAnalyzer(source=FakeSource(items), sentiment_fn=fake_sentiment)
    agg = analyzer.analyze("ACME")
    assert agg.contradictory is False
    assert agg.sentiment_class == "POSITIVE"


# =============================================================================
# Regression: news_score() unaffected by Part 3 additions
# =============================================================================

def test_news_score_unchanged_by_event_and_taxonomy_additions():
    items = [make_item("Great earnings beat", "reuters.com", 1)]
    analyzer = NewsAnalyzer(source=FakeSource(items), sentiment_fn=fake_sentiment, min_source_credibility=0.4)
    agg = analyzer.analyze("ACME")
    # news_score()'s formula reads only overall_sentiment_score/contradictory/
    # overall_confidence -- none of which Part 3's additions touch.
    base = 50.0 + (agg.overall_sentiment_score / 2.0)
    if agg.contradictory:
        base = 50.0 + (base - 50.0) * 0.25
    expected = 50.0 + (base - 50.0) * (agg.overall_confidence / 100.0)
    assert agg.news_score() == pytest.approx(expected)
