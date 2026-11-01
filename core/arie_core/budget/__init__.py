from __future__ import annotations

import time
from collections.abc import Mapping


class BudgetHandle:
    """Enforces a strict tool-call budget for a single agent run."""

    max_tool_calls: int = 3
    max_tokens: int = 12_000
    max_wall_sec: float = 25.0

    def __init__(
        self,
        *,
        max_tool_calls: int = 3,
        max_tokens: int = 12_000,
        max_wall_sec: float = 25.0,
    ) -> None:
        self.max_tool_calls = max_tool_calls
        self.max_tokens = max_tokens
        self.max_wall_sec = max_wall_sec
        self._used_calls = 0
        self._used_tokens = 0.0
        self._seen_tool_calls: set[str] = set()
        self._started_at = time.monotonic()

    def _check_wall_budget(self) -> None:
        if time.monotonic() - self._started_at > self.max_wall_sec:
            raise RuntimeError("wall-clock budget exhausted")

    @staticmethod
    def format_bats_status(
        *,
        used_calls: int,
        used_tokens: float,
        elapsed: float,
        max_calls: int,
        max_tokens: int,
        max_wall_sec: float,
    ) -> str:
        return (
            "BATS: "
            f"calls={used_calls}/{max_calls} "
            f"tokens={int(used_tokens)}/{max_tokens} "
            f"wall={elapsed:.2f}s/{max_wall_sec:.1f}s"
        )

    def reserve(self, amount: float) -> None:
        self._check_wall_budget()
        if amount < 0:
            raise ValueError("reserved amount must be non-negative")
        if self._used_tokens + amount > self.max_tokens:
            raise RuntimeError("token budget exhausted")
        self._used_tokens += amount

    def consume(self, amount: float) -> None:
        self.reserve(amount)

    def record_tool_call(self, name: str, *, args: Mapping[str, object] | None = None) -> str:
        del args
        self._check_wall_budget()
        normalized = str(name)
        if not normalized:
            raise ValueError("tool call name must not be empty")
        if normalized in self._seen_tool_calls:
            raise ValueError(f"duplicate tool call rejected: {normalized}")
        self._seen_tool_calls.add(normalized)
        self._used_calls += 1
        if self._used_calls > self.max_tool_calls:
            raise RuntimeError("tool call budget exhausted")
        return normalized

    @property
    def remaining_tokens(self) -> float:
        return self.max_tokens - self._used_tokens

    @property
    def remaining_calls(self) -> int:
        return self.max_tool_calls - self._used_calls

    def status(self) -> str:
        elapsed = time.monotonic() - self._started_at
        return self.format_bats_status(
            used_calls=self._used_calls,
            used_tokens=self._used_tokens,
            elapsed=elapsed,
            max_calls=self.max_tool_calls,
            max_tokens=self.max_tokens,
            max_wall_sec=self.max_wall_sec,
        )


__all__ = ["BudgetHandle"]
