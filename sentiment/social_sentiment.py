"""
Social Media Sentiment module.

Treated strictly as a SECONDARY signal (10% default weight in the AI signal
engine) -- never proof of anything by itself, and never allowed to push a
trade through on its own. This module measures:

  * sentiment (lexicon-based, same VADER approach as news for consistency)
  * mention volume
  * sentiment acceleration (recent window vs. prior window)
  * positive/negative ratio
  * abnormal activity (volume spike vs. the symbol's own recent baseline)
  * bot/spam-like behavior (near-duplicate text, templated phrasing)
  * potential manipulation (many near-duplicate posts + volume spike +
    one-sided sentiment, i.e. a pump-and-dump-style signature)
  * source credibility (bounded by how anonymous the platform is by default)

Data source is pluggable via `SocialSourceAdapter`. The default adapter
queries Reddit's public, keyless JSON search endpoint (best-effort, heavily
rate-limited, no guarantee of coverage). Swap in a licensed X/Twitter or
StockTwits API adapter later without touching the scoring logic below.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from typing import Callable, Dict, List, Optional

import numpy as np


@dataclass
class SocialPost:
    symbol: str
    text: str
    author: str
    source: str            # e.g. "reddit"
    created_at: datetime
    engagement: int = 0     # upvotes/likes, if available
    sentiment_score: float = 0.0
    sentiment_label: str = "Neutral"


@dataclass
class SocialAggregate:
    symbol: str
    posts: List[SocialPost]
    mention_volume: int
    sentiment_mean: float          # -1..1
    positive_ratio: float
    negative_ratio: float
    sentiment_acceleration: float  # change in mean sentiment, recent vs prior window
    abnormal_activity: bool
    bot_like_fraction: float       # 0-1
    manipulation_suspected: bool
    credibility_factor: float      # 0-1, discount applied for anonymous/low-quality sources
    data_available: bool = True    # explicit flag (spec section 6): False means no reliable
                                    # social data could be obtained for this symbol/window --
                                    # NOT the same as "checked and found neutral." Callers
                                    # (strategy/signal_engine.py) must treat data_available=False
                                    # the same as social=None: excluded from scoring, never
                                    # assumed bullish/neutral.

    def social_score(self) -> float:
        """0-100 scale, 50 = neutral. Manipulation suspicion or a bot-heavy
        sample collapses the score toward neutral regardless of how bullish
        or bearish the raw sentiment looks. Callers must check
        `data_available` BEFORE calling this -- see the class docstring for
        `data_available`; a 50.0 returned here for zero mentions is a
        neutral SCORE for "no signal", not a claim that data was checked
        and genuinely found neutral, and the signal engine never scores it
        as an independent vote in that case (see `social_ok` gating in
        strategy/signal_engine.py)."""
        if not self.data_available or self.mention_volume == 0:
            return 50.0
        base = 50.0 + (self.sentiment_mean * 50.0)
        base = float(np.clip(base, 0, 100))
        discount = self.credibility_factor * (1 - self.bot_like_fraction)
        if self.manipulation_suspected:
            discount *= 0.3
        return float(50.0 + (base - 50.0) * discount)


class SocialSourceAdapter:
    def fetch(self, symbol: str, query: Optional[str] = None, limit: int = 100) -> List[SocialPost]:
        raise NotImplementedError


class RedditSocialSourceAdapter(SocialSourceAdapter):
    """Best-effort, keyless Reddit search. No credentials required, but
    Reddit may rate-limit or block anonymous requests -- callers should
    treat an empty result as 'no data', not as 'confirmed no chatter'."""

    def fetch(self, symbol: str, query: Optional[str] = None, limit: int = 100) -> List[SocialPost]:
        import urllib.request
        import json as _json
        from urllib.parse import quote

        q = quote(query or symbol)
        url = f"https://www.reddit.com/search.json?q={q}&sort=new&limit={min(limit, 100)}"
        req = urllib.request.Request(url, headers={"User-Agent": "trading-system-sentiment/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = _json.loads(resp.read().decode("utf-8"))
        except Exception:
            return []

        posts: List[SocialPost] = []
        for child in data.get("data", {}).get("children", []):
            d = child.get("data", {})
            created = datetime.fromtimestamp(d.get("created_utc", 0), tz=timezone.utc)
            text = f"{d.get('title', '')}. {d.get('selftext', '')}".strip()
            if not text:
                continue
            posts.append(
                SocialPost(
                    symbol=symbol,
                    text=text[:500],
                    author=d.get("author", "unknown"),
                    source="reddit",
                    created_at=created,
                    engagement=int(d.get("score", 0) or 0),
                )
            )
        return posts


def _text_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


class SocialSentimentAnalyzer:
    def __init__(
        self,
        source: Optional[SocialSourceAdapter] = None,
        min_mentions_for_signal: int = 25,
        spam_bot_score_threshold: float = 0.6,
        sentiment_fn: Optional[Callable[[str], float]] = None,
        lookback_hours: int = 48,
    ):
        self.source = source or RedditSocialSourceAdapter()
        self.min_mentions_for_signal = min_mentions_for_signal
        self.spam_bot_score_threshold = spam_bot_score_threshold
        self.lookback_hours = lookback_hours
        self._sentiment_fn = sentiment_fn or self._default_sentiment_fn

    @staticmethod
    def _default_sentiment_fn(text: str) -> float:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        analyzer = SentimentIntensityAnalyzer()
        return analyzer.polarity_scores(text)["compound"]

    def _bot_like_fraction(self, posts: List[SocialPost]) -> float:
        """Heuristic only: flags near-duplicate text and posts authored by
        the same handful of accounts as bot/spam-like. This is NOT a
        reliable bot-detection system -- it exists to discount obviously
        templated/coordinated activity, not to make definitive claims about
        any individual account."""
        if len(posts) < 3:
            return 0.0
        texts = [p.text.lower().strip() for p in posts]
        dup_flags = [False] * len(texts)
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                if _text_similarity(texts[i], texts[j]) > 0.85:
                    dup_flags[i] = True
                    dup_flags[j] = True
        dup_fraction = sum(dup_flags) / len(texts)

        author_counts = Counter(p.author for p in posts)
        most_common_count = author_counts.most_common(1)[0][1] if author_counts else 0
        concentration_fraction = most_common_count / len(posts)

        return float(np.clip(max(dup_fraction, concentration_fraction - 0.3), 0, 1))

    def analyze(
        self,
        symbol: str,
        historical_avg_mentions: Optional[float] = None,
        credibility_factor: float = 0.5,
    ) -> SocialAggregate:
        """
        historical_avg_mentions: the symbol's normal daily mention count, if
        known (e.g. from a prior rolling average you maintain). Used for
        abnormal-activity detection. Without it, abnormal activity is
        flagged only via a conservative absolute threshold.
        """
        raw_posts = self.source.fetch(symbol)
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(hours=self.lookback_hours)
        posts = [p for p in raw_posts if p.created_at >= window_start]

        for p in posts:
            p.sentiment_score = self._sentiment_fn(p.text)
            p.sentiment_label = (
                "Positive" if p.sentiment_score > 0.2 else ("Negative" if p.sentiment_score < -0.2 else "Neutral")
            )

        mention_volume = len(posts)
        if mention_volume == 0:
            # Explicitly UNAVAILABLE, not "checked and neutral": could mean
            # no chatter exists, or the source (e.g. Reddit's keyless search)
            # was rate-limited/blocked -- either way, never assume bullish
            # or neutral from this state (spec section 6).
            return SocialAggregate(
                symbol=symbol, posts=[], mention_volume=0, sentiment_mean=0.0,
                positive_ratio=0.0, negative_ratio=0.0, sentiment_acceleration=0.0,
                abnormal_activity=False, bot_like_fraction=0.0,
                manipulation_suspected=False, credibility_factor=credibility_factor,
                data_available=False,
            )

        sentiments = np.array([p.sentiment_score for p in posts])
        sentiment_mean = float(sentiments.mean())
        positive_ratio = float((sentiments > 0.2).mean())
        negative_ratio = float((sentiments < -0.2).mean())

        midpoint = window_start + (now - window_start) / 2
        recent = [p for p in posts if p.created_at >= midpoint]
        prior = [p for p in posts if p.created_at < midpoint]
        recent_mean = float(np.mean([p.sentiment_score for p in recent])) if recent else sentiment_mean
        prior_mean = float(np.mean([p.sentiment_score for p in prior])) if prior else sentiment_mean
        sentiment_acceleration = recent_mean - prior_mean

        if historical_avg_mentions and historical_avg_mentions > 0:
            abnormal_activity = mention_volume > historical_avg_mentions * 3
        else:
            abnormal_activity = mention_volume > max(self.min_mentions_for_signal * 4, 100)

        bot_like_fraction = self._bot_like_fraction(posts)
        one_sided = max(positive_ratio, negative_ratio) > 0.85
        manipulation_suspected = bool(
            abnormal_activity and one_sided and bot_like_fraction > self.spam_bot_score_threshold
        )

        return SocialAggregate(
            symbol=symbol,
            posts=posts,
            mention_volume=mention_volume,
            sentiment_mean=sentiment_mean,
            positive_ratio=positive_ratio,
            negative_ratio=negative_ratio,
            sentiment_acceleration=sentiment_acceleration,
            abnormal_activity=abnormal_activity,
            bot_like_fraction=bot_like_fraction,
            manipulation_suspected=manipulation_suspected,
            credibility_factor=credibility_factor,
            data_available=True,
        )
