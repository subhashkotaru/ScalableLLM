"""
API response cache for external tool calls (Google Places, Directions, Weather, etc.).
Caches by hash(api_name + sorted params). TTL: 30 minutes.
Tracks hit/miss counts and reports reduction rate.
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from threading import Lock

logger = logging.getLogger(__name__)

TTL_SECONDS = 30 * 60  # 30 minutes


@dataclass
class _Entry:
    value: object
    expires_at: float


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total else 0.0

    def reduction_pct(self) -> float:
        """Percentage of API calls avoided due to cache hits."""
        return self.hit_rate * 100

    def report(self) -> str:
        if self.total == 0:
            return "Cache: no calls made yet."
        return (
            f"Cache: {self.hits} hits / {self.misses} misses / {self.total} total — "
            f"cache reduced API calls by {self.reduction_pct():.1f}%"
        )


class APICache:
    def __init__(self, ttl: int = TTL_SECONDS):
        self._store: dict[str, _Entry] = {}
        self._lock = Lock()
        self._ttl = ttl
        self.stats = CacheStats()

    # ── Key construction ────────────────────────────────────────────────────

    @staticmethod
    def make_key(api_name: str, params: dict) -> str:
        """Stable hash over api_name + sorted params (handles nested objects)."""
        canonical = json.dumps(
            {"api": api_name, "params": params},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(canonical.encode()).hexdigest()

    # ── Core get/set ────────────────────────────────────────────────────────

    def get(self, key: str) -> tuple[bool, object]:
        """Return (hit, value). Expired entries count as misses."""
        with self._lock:
            entry = self._store.get(key)
            if entry and time.monotonic() < entry.expires_at:
                self.stats.hits += 1
                logger.debug("Cache HIT  key=%s…", key[:12])
                return True, entry.value
            if entry:
                del self._store[key]   # evict expired
            self.stats.misses += 1
            logger.debug("Cache MISS key=%s…", key[:12])
            return False, None

    def set(self, key: str, value: object) -> None:
        with self._lock:
            self._store[key] = _Entry(
                value=value,
                expires_at=time.monotonic() + self._ttl,
            )

    # ── Convenience wrapper ─────────────────────────────────────────────────

    def cached_call(self, api_name: str, params: dict, fn) -> object:
        """
        Look up cache; on miss call fn(**params), store result, return it.
        fn should be a callable that accepts **params and returns a JSON-serialisable object.
        """
        key = self.make_key(api_name, params)
        hit, value = self.get(key)
        if hit:
            return value
        result = fn(**params)
        self.set(key, result)
        return result

    # ── Housekeeping ────────────────────────────────────────────────────────

    def evict_expired(self) -> int:
        """Remove all expired entries. Returns count removed."""
        now = time.monotonic()
        with self._lock:
            expired = [k for k, e in self._store.items() if now >= e.expires_at]
            for k in expired:
                del self._store[k]
        return len(expired)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def report(self) -> str:
        return self.stats.report()
