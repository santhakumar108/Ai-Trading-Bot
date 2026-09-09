"""
News Analysis module.

Pipeline, per the spec, for every important news item:
  1. Identify the company (symbol match) and estimate RELEVANCE -- does the
     text actually mention the company, or only match the search query
     loosely (an aggregator match on an unrelated result)?
  2. Determine source credibility (configurable whitelist/score map).
  3. Determine whether the news is new or already priced in (age check).
  4. Detect near-duplicate/syndicated copies of the same underlying story
     (NOVELTY) -- a wire story republished by five outlets is one piece of
     news, not five independent confirmations.
  5. Classify sentiment into 5 buckets using a lexicon-based scorer
     (VADER) -- interpretable, no API key, no opaque LLM call required.
  6. Estimate possible market impact (function of sentiment strength,
     credibility, relevance, and recency).
  7. Detect contradictory reports (opposite-sign sentiment from different
     sources within a time window).
  8. Assign a confidence score.

Every `NewsItem` therefore carries: source, published_at, headline, url,
symbol (ticker/company mapping), credibility, relevance, sentiment,
novelty, and estimated_impact.

The default data source is free RSS feeds (Yahoo Finance / Google News),
fetched through a small adapter interface (`NewsSourceAdapter`) so a paid
news API can be dropped in later without changing any scoring logic below.

Hard rule enforced here: a single headline, however strong, is capped in
both impact and confidence -- corroboration from more than one credible
source is required to reach a high-confidence news signal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from typing import Callable, Dict, List, Optional

import numpy as np

# Default per-domain credibility map. 0 = ignore entirely, 1 = fully credible.
# This is intentionally a simple, editable table rather than a black box --
# extend it via NewsAnalyzer(credibility_map=...).
DEFAULT_CREDIBILITY_MAP: Dict[str, float] = {
    "reuters.com": 0.95,
    "bloomberg.com": 0.95,
    "wsj.com": 0.9,
    "ft.com": 0.9,
    "economictimes.indiatimes.com": 0.8,
    "moneycontrol.com": 0.75,
    "livemint.com": 0.75,
    "cnbc.com": 0.8,
    "finance.yahoo.com": 0.7,
    "news.google.com": 0.5,      # aggregator; underlying source unknown
    "seekingalpha.com": 0.6,
    "reddit.com": 0.2,
    "twitter.com": 0.15,
    "x.com": 0.15,
    "unknown": 0.3,
}


@dataclass
class NewsItem:
    symbol: str
    headline: str
    source_domain: str
    published_at: datetime
    url: str = ""
    summary: str = ""
    # populated by NewsAnalyzer.score_item:
    credibility: float = 0.0
    relevance: float = 1.0            # 0-1, how likely this article is actually ABOUT `symbol`
                                       # (vs. e.g. an aggregator match on an unrelated company)
    sentiment_label: str = "Neutral"
    sentiment_score: float = 0.0      # -1.0 .. +1.0 (compound VADER-style score)
    is_stale: bool = False
    confidence: float = 0.0
    estimated_impact: float = 0.0     # 0-100
    novelty: float = 1.0              # 0-1, 1.0 = first report of this story, lower = a
                                       # re-report/syndicated copy of an already-seen story
    is_duplicate: bool = False        # True if collapsed into an earlier item as a near-duplicate/
                                       # syndicated copy -- excluded from scoring, kept for audit
    # Spec Part 3: event classification + named sentiment taxonomy. ADDED
    # alongside sentiment_label/sentiment_score above (kept unchanged) --
    # same "add, don't touch" pattern as indicators/technical.py's family
    # scores and fundamentals/fundamental_analysis.py's dimension scores.
    event_type: str = "OTHER"         # see classify_event_type() below
    sentiment_class: str = "UNKNOWN"  # POSITIVE|NEGATIVE|NEUTRAL|UNKNOWN -- "UNKNOWN" only when
                                       # there was no text to score at all; MIXED does not apply at
                                       # the single-item level (see classify_sentiment_taxonomy).
    affected_timeframe: str = "SHORT_TERM"  # see _affected_timeframe() below


@dataclass
class NewsAggregate:
    symbol: str
    items: List[NewsItem]
    overall_sentiment_score: float     # -100 .. +100
    overall_confidence: float          # 0-100
    contradictory: bool
    num_credible_sources: int
    # Spec Part 3's POSITIVE/NEGATIVE/NEUTRAL/MIXED/UNKNOWN taxonomy at the
    # AGGREGATE level: "MIXED" tracks the EXISTING `contradictory` flag
    # above (no new contradiction logic), "UNKNOWN" means zero credible
    # sources, otherwise derived from `overall_sentiment_score`. Computed
    # once in `analyze()` and stored (not a property) so it's also
    # available on directly-constructed test/aggregate fixtures.
    sentiment_class: str = "UNKNOWN"

    def news_score(self) -> float:
        """
        Map to the 0-100 scale used by the signal engine (50 = neutral).
        Confidence-gated: low confidence or contradictory reports pull the
        score back toward neutral rather than letting a shaky signal swing
        the overall decision.
        """
        base = 50.0 + (self.overall_sentiment_score / 2.0)  # -100..100 -> 0..100
        base = float(np.clip(base, 0, 100))
        if self.contradictory:
            base = 50.0 + (base - 50.0) * 0.25   # heavily dampened
        confidence_factor = self.overall_confidence / 100.0
        return float(50.0 + (base - 50.0) * confidence_factor)


class NewsSourceAdapter:
    """Interface for pulling raw headlines for a symbol. Swap the default
    RSS implementation for a paid API by subclassing and passing an
    instance into NewsAnalyzer(source=...)."""

    def fetch(self, symbol: str, company_name: Optional[str] = None) -> List[NewsItem]:
        raise NotImplementedError


class RSSNewsSourceAdapter(NewsSourceAdapter):
    """Free, no-key news source using Google News' public RSS search feed.
    Network access is required only when `fetch` is actually called; unit
    tests inject a fake adapter instead."""

    def fetch(self, symbol: str, company_name: Optional[str] = None) -> List[NewsItem]:
        import feedparser
        from urllib.parse import quote

        query = quote(company_name or symbol)
        url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:US"
        parsed = feedparser.parse(url)
        items: List[NewsItem] = []
        for entry in parsed.entries[:25]:
            published = datetime.now(timezone.utc)
            if getattr(entry, "published_parsed", None):
                published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
            source_domain = "news.google.com"
            if hasattr(entry, "source") and getattr(entry.source, "href", None):
                try:
                    from urllib.parse import urlparse
                    source_domain = urlparse(entry.source.href).netloc.replace("www.", "")
                except Exception:
                    pass
            items.append(
                NewsItem(
                    symbol=symbol,
                    headline=entry.title,
                    source_domain=source_domain,
                    published_at=published,
                    url=getattr(entry, "link", ""),
                    summary=getattr(entry, "summary", ""),
                )
            )
        return items


_LABELS = ["Strong Negative", "Negative", "Neutral", "Positive", "Strong Positive"]


def _classify_sentiment(compound: float) -> str:
    if compound <= -0.6:
        return "Strong Negative"
    if compound <= -0.2:
        return "Negative"
    if compound < 0.2:
        return "Neutral"
    if compound < 0.6:
        return "Positive"
    return "Strong Positive"


# Spec Part 3: event classification, ordered MOST-SPECIFIC first so a rare,
# high-signal category (fraud, regulatory action) isn't shadowed by a
# generic one (earnings) that happens to share a common word. Simple
# keyword/phrase matching -- explainable, no black box, same principle
# already stated in this module's docstring for its lexicon-based sentiment
# scorer. First category with a matching phrase wins; "OTHER" if none match.
_EVENT_KEYWORDS: List[tuple] = [
    ("FRAUD_ALLEGATION", ["fraud", "scam", "whistleblower", "accounting irregularit", "siphon", "embezzle"]),
    ("REGULATORY_ACTION", ["sebi ", "rbi probe", "regulatory action", "regulator ", "show cause notice",
                            "compliance violation", "penalty imposed", "banned by", "market regulator"]),
    ("LEGAL_ISSUE", ["lawsuit", "litigation", "court order", "legal action", "sues ", "sued by", "sued for"]),
    ("MANAGEMENT_CHANGE", ["steps down", "resigns as", "resignation of", "appoints new", "appointed as ceo",
                            "appointed as md", "appointed as cfo", "new ceo", "new md", "new cfo",
                            "management change", "quits as"]),
    ("ORDER_LOSS", ["loses order", "loses contract", "contract terminated", "contract cancelled",
                     "order cancelled", "order lost"]),
    ("ORDER_WIN", ["wins order", "bags order", "secures order", "wins contract", "bags contract",
                    "awarded contract", "order win", "receives order", "clinches order",
                    "wins bid", "bags bid", "wins deal", "bags deal"]),
    # Real headline pattern (found via a live-data check, not synthetic):
    # verb and object separated by an amount/description --
    # "Wins INR1.23 Billion ... Automation Bid" -- plain substring matching
    # misses this; a bounded regex catches "wins/bags/secures ... bid/
    # contract/order/deal" without the two words needing to be adjacent.
    ("ACQUISITION", ["acquires", "to acquire", "acquisition of", "buys stake", "acquired by", "buyout of"]),
    ("MERGER", ["merger", "to merge", "amalgamation", "merges with"]),
    ("DEBT", ["credit rating downgrade", "credit rating upgrade", "rating downgraded", "rating upgraded",
              "debt restructuring", "loan default", "defaults on", "downgrades rating"]),
    ("PROMOTER_ACTIVITY", ["promoter pledge", "promoter stake", "pledged shares", "promoter holding",
                            "promoters sell", "promoters buy"]),
    ("INSIDER_ACTIVITY", ["insider trading", "bulk deal", "block deal"]),
    ("DIVIDEND", ["dividend"]),
    ("BUYBACK", ["buyback", "buy-back", "share repurchase"]),
    ("SPLIT", ["stock split", "share split"]),
    ("BONUS", ["bonus share", "bonus issue"]),
    ("CORPORATE_ACTION", ["rights issue", "demerger", "spin-off", "spinoff"]),
    ("EARNINGS", ["quarterly result", "q1 result", "q2 result", "q3 result", "q4 result", "net profit",
                   "earnings beat", "earnings miss", "reports profit", "reports loss", "results announced",
                   "profit rises", "profit falls", "profit jumps", "profit drops"]),
    ("GUIDANCE", ["guidance", "outlook raised", "outlook cut", "forecast raised", "forecast cut",
                   "raises outlook", "cuts outlook"]),
    ("MACRO_IMPACT", ["repo rate", "rbi rate", "inflation", "gdp growth", "union budget",
                       "interest rate hike", "interest rate cut", "rate hike", "rate cut"]),
    ("GEOPOLITICAL_EVENT", ["geopolitical", "trade war", "sanctions", "tariff", " war ", "conflict escalat"]),
    ("SECTOR_EVENT", ["sector-wide", "industry-wide", "industry wide", "sector wide"]),
]

# Proximity regex patterns, checked at the SAME priority position as their
# category's phrase list above (see classify_event_type) -- for real
# headline patterns where the verb and object aren't adjacent (e.g. "Wins
# INR1.23 Billion ... Automation Bid"), which plain substring matching on
# `_EVENT_KEYWORDS` misses. Bounded to 80 characters between verb and
# object so this doesn't match across an entire long headline/summary.
_EVENT_REGEX: Dict[str, re.Pattern] = {
    "ORDER_WIN": re.compile(r"\b(wins?|bags?|secures?|clinch(?:es)?)\b.{0,80}\b(bid|contract|order|deal)s?\b"),
    "ORDER_LOSS": re.compile(r"\b(loses?|lost|terminat\w*|cancell?ed)\b.{0,80}\b(bid|contract|order|deal)s?\b"),
}

# Spec Part 3: affected-timeframe lookup, driven by event_type -- a small,
# explainable mapping, not a separate classifier.
_TIMEFRAME_BY_EVENT: Dict[str, str] = {
    "DIVIDEND": "IMMEDIATE_CORPORATE_ACTION", "BUYBACK": "IMMEDIATE_CORPORATE_ACTION",
    "SPLIT": "IMMEDIATE_CORPORATE_ACTION", "BONUS": "IMMEDIATE_CORPORATE_ACTION",
    "CORPORATE_ACTION": "IMMEDIATE_CORPORATE_ACTION",
    "EARNINGS": "QUARTER", "GUIDANCE": "QUARTER",
    "MACRO_IMPACT": "MARKET_WIDE_ONGOING", "GEOPOLITICAL_EVENT": "MARKET_WIDE_ONGOING",
    "SECTOR_EVENT": "MARKET_WIDE_ONGOING",
}


def classify_event_type(headline: str, summary: str = "") -> str:
    """Spec Part 3: ordered keyword match against `_EVENT_KEYWORDS` --
    first category with a matching phrase (or, for a few categories with
    a common non-adjacent real-headline pattern, a bounded proximity regex
    from `_EVENT_REGEX`) wins. `"OTHER"` if nothing matches (never a guess)."""
    text = f" {headline} {summary} ".lower()
    for event_type, phrases in _EVENT_KEYWORDS:
        if any(phrase in text for phrase in phrases):
            return event_type
        pattern = _EVENT_REGEX.get(event_type)
        if pattern is not None and pattern.search(text):
            return event_type
    return "OTHER"


def classify_sentiment_taxonomy(compound: float) -> str:
    """Spec Part 3's named taxonomy, ITEM level: POSITIVE/NEGATIVE/NEUTRAL
    only -- MIXED and UNKNOWN don't apply to a single VADER compound score
    (see NewsItem.sentiment_class's docstring / NewsAggregate.sentiment_class
    for where those two apply)."""
    if compound >= 0.2:
        return "POSITIVE"
    if compound <= -0.2:
        return "NEGATIVE"
    return "NEUTRAL"


def _affected_timeframe(event_type: str) -> str:
    return _TIMEFRAME_BY_EVENT.get(event_type, "SHORT_TERM")


class NewsAnalyzer:
    def __init__(
        self,
        source: Optional[NewsSourceAdapter] = None,
        credibility_map: Optional[Dict[str, float]] = None,
        min_source_credibility: float = 0.4,
        max_headline_age_hours: int = 48,
        contradiction_window_hours: int = 24,
        sentiment_fn: Optional[Callable[[str], float]] = None,
    ):
        self.source = source or RSSNewsSourceAdapter()
        self.credibility_map = credibility_map or DEFAULT_CREDIBILITY_MAP
        self.min_source_credibility = min_source_credibility
        self.max_headline_age_hours = max_headline_age_hours
        self.contradiction_window_hours = contradiction_window_hours
        self._sentiment_fn = sentiment_fn or self._default_sentiment_fn

    @staticmethod
    def _default_sentiment_fn(text: str) -> float:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        analyzer = SentimentIntensityAnalyzer()
        return analyzer.polarity_scores(text)["compound"]

    def _credibility_for(self, domain: str) -> float:
        return self.credibility_map.get(domain, self.credibility_map.get("unknown", 0.3))

    @staticmethod
    def _relevance_for(item: NewsItem, symbol: str, company_name: Optional[str]) -> float:
        """Heuristic relevance: does the headline/summary actually mention
        the company, or only the raw ticker (which an aggregator search can
        loosely match on unrelated results)? This is deliberately simple
        and explainable, not an NLP entity-linking model -- a genuine
        ambiguous case (e.g. a generic word that happens to equal a ticker)
        is scored as uncertain (0.5), never as confidently relevant."""
        text = f"{item.headline} {item.summary}".lower()
        ticker_root = symbol.split(".")[0].lower()
        candidates = [c.lower() for c in ([company_name] if company_name else []) + [ticker_root]]
        if any(c and c in text for c in candidates):
            return 1.0
        return 0.5  # matched by the search query itself, but not confirmed in the text

    def _mark_duplicates_and_novelty(self, items: List[NewsItem]) -> None:
        """Detects near-duplicate / syndicated copies of the same story
        (same headline reworded, republished by another outlet, etc.) so a
        single event doesn't masquerade as multiple independent
        corroborating sources. The earliest copy of a story is kept
        (novelty=1.0); later near-duplicates are flagged `is_duplicate` and
        excluded from scoring (novelty=0.0), but kept in `NewsAggregate.items`
        for audit -- nothing is deleted."""
        ordered = sorted(items, key=lambda i: i.published_at)
        kept: List[NewsItem] = []
        for item in ordered:
            is_dup = False
            for prior in kept:
                similarity = SequenceMatcher(None, item.headline.lower().strip(), prior.headline.lower().strip()).ratio()
                if similarity > 0.80:
                    is_dup = True
                    break
            item.is_duplicate = is_dup
            item.novelty = 0.0 if is_dup else 1.0
            if not is_dup:
                kept.append(item)

    def score_item(self, item: NewsItem, now: Optional[datetime] = None, symbol: str = "", company_name: Optional[str] = None) -> NewsItem:
        now = now or datetime.now(timezone.utc)
        item.credibility = self._credibility_for(item.source_domain)
        item.relevance = self._relevance_for(item, symbol or item.symbol, company_name)
        age_hours = max((now - item.published_at).total_seconds() / 3600.0, 0.0)
        item.is_stale = age_hours > self.max_headline_age_hours

        compound = self._sentiment_fn(f"{item.headline}. {item.summary}")
        item.sentiment_score = compound
        item.sentiment_label = _classify_sentiment(compound)

        # Spec Part 3: event classification + named sentiment taxonomy.
        item.event_type = classify_event_type(item.headline, item.summary)
        item.affected_timeframe = _affected_timeframe(item.event_type)
        has_text = bool((item.headline or "").strip() or (item.summary or "").strip())
        item.sentiment_class = classify_sentiment_taxonomy(compound) if has_text else "UNKNOWN"

        recency_factor = max(0.0, 1.0 - age_hours / (self.max_headline_age_hours * 2))
        item.confidence = float(np.clip(item.credibility * item.relevance * (0.5 + 0.5 * recency_factor), 0, 1)) * 100

        impact = abs(compound) * item.credibility * item.relevance * (0.4 + 0.6 * recency_factor)
        item.estimated_impact = float(np.clip(impact, 0, 1)) * 100
        return item

    def analyze(self, symbol: str, company_name: Optional[str] = None) -> NewsAggregate:
        raw_items = self.source.fetch(symbol, company_name)
        now = datetime.now(timezone.utc)
        scored = [self.score_item(i, now, symbol=symbol, company_name=company_name) for i in raw_items]
        self._mark_duplicates_and_novelty(scored)

        # Drop items from sources below the credibility floor, stale items,
        # low-relevance matches, and duplicate/syndicated copies of a story
        # already counted -- they still show up in `items` for audit, but
        # never inflate the score or the "how many independent sources"
        # count.
        credible = [
            i for i in scored
            if i.credibility >= self.min_source_credibility
            and not i.is_stale
            and not i.is_duplicate
            and i.relevance >= 0.5
        ]

        if not credible:
            return NewsAggregate(
                symbol=symbol,
                items=scored,
                overall_sentiment_score=0.0,
                overall_confidence=0.0,
                contradictory=False,
                num_credible_sources=0,
                sentiment_class="UNKNOWN",
            )

        # Contradiction detection: within the contradiction window, do we have
        # both a clearly positive and a clearly negative credible item from
        # *different* sources?
        window_start = now - timedelta(hours=self.contradiction_window_hours)
        recent = [i for i in credible if i.published_at >= window_start]
        pos_sources = {i.source_domain for i in recent if i.sentiment_score >= 0.2}
        neg_sources = {i.source_domain for i in recent if i.sentiment_score <= -0.2}
        contradictory = len(pos_sources) > 0 and len(neg_sources) > 0 and pos_sources != neg_sources

        # Weighted average sentiment (weight = credibility * confidence),
        # scaled to -100..100.
        weights = np.array([i.credibility * (i.confidence / 100.0) for i in credible])
        sentiments = np.array([i.sentiment_score for i in credible])
        weighted_sentiment = float(np.average(sentiments, weights=weights)) * 100

        num_credible_sources = len({i.source_domain for i in credible})
        # "Never trade solely on one headline": with only a single credible
        # source, cap confidence hard regardless of how strong the sentiment is.
        base_confidence = float(np.mean([i.confidence for i in credible]))
        if num_credible_sources < 2:
            base_confidence = min(base_confidence, 35.0)
        if contradictory:
            base_confidence *= 0.5

        # Spec Part 3, AGGREGATE level: "MIXED" tracks the contradictory
        # flag computed above (no new logic); otherwise derived from the
        # weighted sentiment score. "UNKNOWN" is reserved for the
        # zero-credible-sources case above.
        if contradictory:
            aggregate_sentiment_class = "MIXED"
        else:
            aggregate_sentiment_class = classify_sentiment_taxonomy(weighted_sentiment / 100.0)

        return NewsAggregate(
            symbol=symbol,
            items=scored,
            overall_sentiment_score=weighted_sentiment,
            overall_confidence=float(np.clip(base_confidence, 0, 100)),
            contradictory=contradictory,
            num_credible_sources=num_credible_sources,
            sentiment_class=aggregate_sentiment_class,
        )
