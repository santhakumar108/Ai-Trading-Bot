"""
Configurable NSE-universe scanner (spec section 19).

Wraps `PaperTradingEngine.scan_symbol` to scan a whole universe -- by
default `config.universe.symbols` (NIFTY 50 / Next 50 / whatever
`config.yaml` actually sets), NEVER a hardcoded stock list like the
`DEMO_UP`/`DEMO_DOWN`/... names used only in `examples/offline_demo.py`'s
fully-offline demo. Adds three things `PaperTradingEngine.scan_universe`
does not do on its own:

  * BATCHING + a pause between batches ("don't scan thousands blindly if
    API limits unreliable" -- a free/shared market-data API can rate-limit
    or degrade under a burst of back-to-back requests).
  * A short-TTL result CACHE so a dashboard or CLI polling loop that
    re-scans the same universe within a few tens of seconds doesn't
    refetch/rescoring identical data for no reason.
  * Per-symbol fault isolation: one symbol raising an unexpected exception
    is logged and skipped, never allowed to abort the whole cycle (spec
    section 22: a single data source's failure must never take the system
    down or force a trade).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from paper_trading.engine import PaperTradingEngine, ScanResult

logger = logging.getLogger(__name__)


@dataclass
class ScanCycleResult:
    results: List[ScanResult]
    symbols_requested: int
    symbols_scanned: int     # got a usable ScanResult (may still be NO TRADE)
    symbols_skipped: int     # no data / analysis error -- scan_symbol returned None or raised
    started_at: datetime
    finished_at: datetime
    from_cache: bool = False


class UniverseScanner:
    def __init__(
        self,
        engine: PaperTradingEngine,
        batch_size: Optional[int] = None,
        delay_between_batches_seconds: Optional[float] = None,
        cache_ttl_seconds: Optional[float] = None,
    ):
        """
        Any of `batch_size` / `delay_between_batches_seconds` /
        `cache_ttl_seconds` left as None fall back to
        `engine.config.universe.scan_batch_size` /
        `scan_batch_delay_seconds` / `scan_cache_ttl_seconds` (spec section
        21: configuration-based, not hard-coded) -- pass an explicit value
        only to override the configured default for one scanner instance.
        """
        self.engine = engine
        uc = engine.config.universe
        self.batch_size = max(1, batch_size if batch_size is not None else uc.scan_batch_size)
        self.delay_between_batches_seconds = max(
            0.0, delay_between_batches_seconds if delay_between_batches_seconds is not None
            else uc.scan_batch_delay_seconds,
        )
        self.cache_ttl_seconds = max(
            0.0, cache_ttl_seconds if cache_ttl_seconds is not None else uc.scan_cache_ttl_seconds,
        )
        self._cache: Optional[ScanCycleResult] = None
        self._cache_key: Optional[Tuple] = None

    def scan(
        self,
        symbols: Optional[List[str]] = None,
        sector_map: Optional[Dict[str, str]] = None,
        force_refresh: bool = False,
    ) -> ScanCycleResult:
        """
        `symbols` defaults to `self.engine.config.universe.symbols` -- the
        CONFIGURED universe, never a hardcoded demo list. Scans in batches
        of `self.batch_size` with `self.delay_between_batches_seconds`
        between batches. A cached result for the identical (symbols,
        sector_map) request younger than `self.cache_ttl_seconds` is
        returned instead of re-scanning, unless `force_refresh=True`.
        """
        symbols = list(symbols) if symbols is not None else list(self.engine.config.universe.symbols)
        sector_map = sector_map if sector_map is not None else self.engine.config.universe.sector_map
        cache_key = (tuple(symbols), tuple(sorted((sector_map or {}).items())))

        if not force_refresh and self._cache is not None and self._cache_key == cache_key:
            age = (datetime.now(timezone.utc) - self._cache.finished_at).total_seconds()
            if age < self.cache_ttl_seconds:
                logger.info(
                    "UniverseScanner: serving cached scan (%.0fs old, ttl=%.0fs, %d symbol(s)).",
                    age, self.cache_ttl_seconds, len(symbols),
                )
                cached = self._cache
                return ScanCycleResult(
                    results=cached.results, symbols_requested=cached.symbols_requested,
                    symbols_scanned=cached.symbols_scanned, symbols_skipped=cached.symbols_skipped,
                    started_at=cached.started_at, finished_at=cached.finished_at, from_cache=True,
                )

        started_at = datetime.now(timezone.utc)
        results: List[ScanResult] = []
        skipped = 0

        for batch_start in range(0, len(symbols), self.batch_size):
            batch = symbols[batch_start: batch_start + self.batch_size]
            for symbol in batch:
                try:
                    result = self.engine.scan_symbol(symbol, sector=(sector_map or {}).get(symbol))
                except Exception as exc:
                    logger.warning("UniverseScanner: scan failed for %s (skipping, not aborting the cycle): %s", symbol, exc)
                    result = None
                if result is None:
                    skipped += 1
                else:
                    results.append(result)

            is_last_batch = batch_start + self.batch_size >= len(symbols)
            if not is_last_batch and self.delay_between_batches_seconds > 0:
                time.sleep(self.delay_between_batches_seconds)

        finished_at = datetime.now(timezone.utc)
        cycle = ScanCycleResult(
            results=results, symbols_requested=len(symbols), symbols_scanned=len(results),
            symbols_skipped=skipped, started_at=started_at, finished_at=finished_at, from_cache=False,
        )
        self._cache = cycle
        self._cache_key = cache_key
        return cycle
