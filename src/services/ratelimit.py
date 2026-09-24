"""A small sliding-window rate limiter.

Owner: sara.  Implements the limits in ``config/auth.json`` and the contract §4 -- the
login limiter (FR i, FR lxxviii) and the upload limiter.

Deliberately in-process rather than in Redis: this application is served by a single
gunicorn worker set behind one host, the deployment instructions pin it that way, and a
second moving part during a live demo is a liability. The trade-off is recorded here so it
is a decision rather than an oversight: **if the app is ever scaled to several machines,
these counters must move to a shared store or each machine gets its own allowance.**

The window is measured in real time (``time.monotonic``), so a clock adjustment cannot
grant or deny a request.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Hashable


@dataclass
class LimitResult:
    """The answer, with enough detail for a useful ``429`` and its ``Retry-After``."""

    allowed: bool
    remaining: int
    retry_after_seconds: int

    def __bool__(self) -> bool:  # `if limiter.check(...):`
        return self.allowed


@dataclass
class RateLimiter:
    """N events per ``window_seconds``, separately for each key.

    Keys are caller-chosen -- an IP, a username, or ``f"{user_id}:upload"`` -- so one
    limiter serves every limit in the config by being asked with a different prefix.
    """

    limit: int
    window_seconds: float
    _hits: dict[Hashable, Deque[float]] = field(default_factory=dict, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def check(self, key: Hashable) -> LimitResult:
        """Record an attempt and say whether it is allowed.

        A refused attempt is *not* recorded, so a client that backs off is released on
        schedule rather than being pushed further out by its own retries.
        """
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
        """Remaining allowance, without spending any of it."""
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
        """Clear one key, or everything. A successful sign-in clears that username's key."""
        with self._lock:
            if key is None:
                self._hits.clear()
            else:
                self._hits.pop(key, None)

    def _prune(self, now: float) -> None:
        """Drop keys that have gone quiet, so the counters cannot grow without bound.

        An unbounded dict keyed on client-supplied values is itself a memory-exhaustion
        vector; this makes the limiter's own footprint bounded.
        """
        if len(self._hits) < 512:
            return
        cutoff = now - self.window_seconds
        for key in [k for k, v in self._hits.items() if not v or v[-1] <= cutoff]:
            del self._hits[key]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<RateLimiter {self.limit}/{self.window_seconds:g}s keys={len(self._hits)}>"
