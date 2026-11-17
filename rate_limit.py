"""Per-visitor daily rate limit that protects the free Gemini quota on the public demo.

Counts are kept in memory (shared by all sessions of one running app), so they reset
when the app restarts. Good enough for a demo; production would use Redis or a database.
"""

from __future__ import annotations

import threading
from datetime import date


class DailyRateLimiter:
    def __init__(self, limit_per_day: int):
        self.limit = limit_per_day
        self._counts: dict[tuple[str, date], int] = {}
        self._lock = threading.Lock()

    def remaining(self, visitor: str) -> int:
        with self._lock:
            return max(0, self.limit - self._counts.get((visitor, date.today()), 0))

    def try_consume(self, visitor: str, cost: int = 1) -> bool:
        """Use `cost` runs for this visitor. Returns False (and uses nothing) if not enough left."""
        with self._lock:
            key = (visitor, date.today())
            used = self._counts.get(key, 0)
            if used + cost > self.limit:
                return False
            self._counts[key] = used + cost
            # Forget old days so memory does not grow
            for old in [k for k in self._counts if k[1] != key[1]]:
                del self._counts[old]
            return True

    def refund(self, visitor: str, cost: int = 1) -> None:
        """Give runs back (used when the run failed because of the AI service, not the visitor)."""
        with self._lock:
            key = (visitor, date.today())
            self._counts[key] = max(0, self._counts.get(key, 0) - cost)
