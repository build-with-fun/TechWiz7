"""Sliding-window rate limiter for logins and uploads (limits in config/auth.json).

Counters are kept in memory, which is fine for a single server. With several servers
they would need a shared store.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Hashable


@dataclass
class LimitResult:
    """Result of a check, with what's needed for a 429 and Retry-After."""

    allowed: bool
    remaining: int
    retry_after_seconds: int

    def __bool__(self) -> bool:  # `if limiter.check(...):`
        return self.allowed


@dataclass
class RateLimiter:
    """Allows N events per ``window_seconds`` per key (IP, username, ...)."""

    limit: int
    window_seconds: float
    _hits: dict[Hashable, Deque[float]] = field(default_factory=dict, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def check(self, key: Hashable) -> LimitResult:
        """Record an attempt and return whether it is allowed. Refused attempts aren't counted."""
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            hits = self._hits.get(key)
            if hits is None:
                hits = self._hits[key] = deque()
            while hits and hits[0] <= cutoff:
                hits.popleft()

            if len(hits) >= self.limit:
                retry_after = max(1, int(hits[0] + self.window_seconds - now) + 1)
                return LimitResult(False, 0, retry_after)

            hits.append(now)
            self._prune(now)
            return LimitResult(True, self.limit - len(hits), 0)

    def peek(self, key: Hashable) -> int:
        """Remaining allowance, without using any."""
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            hits = self._hits.get(key)
            if not hits:
                return self.limit
            while hits and hits[0] <= cutoff:
                hits.popleft()
            return max(0, self.limit - len(hits))

    def reset(self, key: Hashable | None = None) -> None:
        """Clear one key, or all of them."""
        with self._lock:
            if key is None:
                self._hits.clear()
            else:
                self._hits.pop(key, None)

    def _prune(self, now: float) -> None:
        """Drop idle keys so the dict doesn't grow forever."""
        if len(self._hits) < 512:
            return
        cutoff = now - self.window_seconds
        for key in [k for k, v in self._hits.items() if not v or v[-1] <= cutoff]:
            del self._hits[key]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<RateLimiter {self.limit}/{self.window_seconds:g}s keys={len(self._hits)}>"
