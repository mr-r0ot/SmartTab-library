"""Shared wall-clock deadline support for a complete fit call."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from smarttab.exceptions import SmartTabError


class TimeLimitExceeded(SmartTabError):
    """Raised when a configured fit deadline has expired."""


@dataclass(slots=True)
class FitDeadline:
    seconds: float = 0.0
    started_at: float = field(init=False)
    ends_at: float | None = field(init=False)

    def __post_init__(self) -> None:
        self.started_at = time.perf_counter()
        self.ends_at = self.started_at + self.seconds if self.seconds > 0 else None

    @property
    def enabled(self) -> bool:
        return self.ends_at is not None

    def elapsed(self) -> float:
        return time.perf_counter() - self.started_at

    def remaining(self) -> float | None:
        if self.ends_at is None:
            return None
        return max(0.0, self.ends_at - time.perf_counter())

    def expired(self) -> bool:
        remaining = self.remaining()
        return remaining is not None and remaining <= 0

    def require(self, stage: str) -> None:
        if self.expired():
            raise TimeLimitExceeded(f"time_limit expired before {stage}")

    def bounded_timeout(self, requested: float | None, reserve: float = 0.0) -> float | None:
        remaining = self.remaining()
        if remaining is None:
            return requested
        available = max(0.0, remaining - reserve)
        if requested is None:
            return available
        return min(float(requested), available)
