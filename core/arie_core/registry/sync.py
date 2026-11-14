from __future__ import annotations

import calendar
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def months_ago(now: datetime, months: int) -> datetime:
    """Subtract calendar months, clamping month-end dates."""
    index = now.year * 12 + now.month - 1 - months
    year, month = divmod(index, 12)
    return now.replace(
        year=year, month=month + 1, day=min(now.day, calendar.monthrange(year, month + 1)[1])
    )


class SyncConfig(BaseModel):
    """Validate unambiguous publication windows and bounded, resumable sync settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: Literal["relative", "year", "month", "since", "range"] = "relative"
    initial_months: int = Field(default=6, ge=1, le=120, strict=True)
    year: int | None = Field(default=None, ge=1991, le=9998, strict=True)
    month: int | None = Field(default=None, ge=1, le=12, strict=True)
    start_month: str | None = Field(default=None, pattern=r"^[0-9]{4}-(0[1-9]|1[0-2])$")
    end_month: str | None = Field(default=None, pattern=r"^[0-9]{4}-(0[1-9]|1[0-2])$")
    batch_size: int = Field(default=20, ge=1, le=100, strict=True)
    max_papers: int = Field(default=100, ge=1, le=10000, strict=True)
    # Library retention: after each sync keep only the newest N papers (None keeps all).
    # Papers with evaluation questions are never removed and do not count toward N.
    keep_papers: int | None = Field(default=None, ge=1, le=100000, strict=True)

    @model_validator(mode="after")
    def validate_window(self) -> SyncConfig:
        expected = {
            "relative": set[str](),
            "year": {"year"},
            "month": {"year", "month"},
            "since": {"start_month"},
            "range": {"start_month", "end_month"},
        }[self.mode]
        present = {
            key
            for key in ("year", "month", "start_month", "end_month")
            if getattr(self, key) is not None
        }
        if present != expected:
            fields = ", ".join(sorted(expected)) or "no dates"
            raise ValueError(f"{self.mode} mode requires only: {fields}")
        for value in (self.start_month, self.end_month):
            if value and not 1991 <= int(value[:4]) <= 9998:
                raise ValueError("Publication years must be between 1991 and 9998")
        if self.start_month and self.end_month and self.start_month > self.end_month:
            raise ValueError("Start month must be on or before end month")
        return self

    def window(self, now: datetime) -> tuple[datetime, datetime]:
        """Resolve inclusive UTC bounds; reject future-only windows and clamp ongoing periods."""
        now = now.astimezone(UTC)
        if self.mode == "relative":
            return months_ago(now, self.initial_months), now
        if self.mode in {"year", "month"}:
            start = datetime(self.year or now.year, self.month or 1, 1, tzinfo=UTC)
        else:
            start = datetime.fromisoformat(f"{self.start_month}-01").replace(tzinfo=UTC)
        if self.mode == "since":
            end = now
        elif self.mode == "year":
            end = start.replace(year=start.year + 1)
        else:
            last = (
                datetime.fromisoformat(f"{self.end_month}-01").replace(tzinfo=UTC)
                if self.end_month
                else start
            )
            end = (
                last.replace(year=last.year + 1, month=1)
                if last.month == 12
                else (last.replace(month=last.month + 1))
            )
        if self.mode != "since":
            end = min(now, end - timedelta(microseconds=1))
        if start > end:
            raise ValueError("Publication window starts in the future")
        return start, end


def papers_to_prune(papers: Iterable[tuple[str, str]], protected: set[str], keep: int) -> list[str]:
    """Paper ids beyond the ``keep`` newest, oldest last; ``protected`` ids are never listed.

    ``papers`` are ``(paper_id, published_at)`` pairs; ties on the date keep the higher id, so
    the choice is the same on every run. Protected papers do not count toward ``keep``.
    """
    ranked = sorted(
        ((published, paper_id) for paper_id, published in papers if paper_id not in protected),
        reverse=True,
    )
    return [paper_id for _, paper_id in ranked[keep:]]
