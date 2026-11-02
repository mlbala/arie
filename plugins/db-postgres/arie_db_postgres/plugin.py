from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from arie_core.events import EventBus
from arie_core.plugin_api import plugin
from arie_db_sqlite.plugin import SQLiteDatabasePlugin


@plugin(slot="database", name="postgres", api=">=1,<2")
class PostgresDatabasePlugin(SQLiteDatabasePlugin):
    """Postgres-compatible database adapter backed by SQLite for local tests.

    The adapter keeps the canonical ARIE schema and exposes a lightweight
    LISTEN/NOTIFY bridge via an in-process EventBus to match the cloud profile
    contract without introducing vendor SDK dependencies.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        path: str | Path | None = None,
        dsn: str | None = None,
    ) -> None:
        candidate = path if path is not None else db_path if db_path is not None else dsn
        if candidate is None:
            candidate = "data/arie-postgres.sqlite3"
        if isinstance(candidate, str) and candidate.startswith("sqlite:///"):
            candidate = candidate.removeprefix("sqlite:///")
        super().__init__(Path(candidate))
        self._event_bus = EventBus()

    def listen(
        self,
        channel: str,
        callback: Callable[[dict[str, object]], object],
    ) -> Callable[[], None]:
        return self._event_bus.subscribe(channel, callback)

    def notify(
        self,
        channel: str,
        payload: Mapping[str, object] | None = None,
    ) -> list[object]:
        return self._event_bus.publish(channel, dict(payload or {}))

    async def notify_async(
        self,
        channel: str,
        payload: Mapping[str, object] | None = None,
    ) -> list[object]:
        return await self._event_bus.publish_async(channel, dict(payload or {}))

    def add_listener(
        self,
        channel: str,
        callback: Callable[[dict[str, object]], object],
    ) -> Callable[[], None]:
        return self.listen(channel, callback)

    def trigger(self, channel: str, payload: Mapping[str, object] | None = None) -> list[object]:
        return self.notify(channel, payload)

    async def execute(
        self,
        ctx: object | None = None,
        query: str | None = None,
        params: Sequence[object] = (),
    ) -> int:
        return await super().execute(ctx, query, params)

    async def fetch(
        self,
        ctx: object | None = None,
        query: str | None = None,
        params: Sequence[object] = (),
    ) -> list[dict[str, object]]:
        return await super().fetch(ctx, query, params)

    async def query(
        self,
        *,
        sql: str,
        params: Sequence[object] | None = None,
    ) -> list[dict[str, Any]]:
        rows = await self.fetch(None, sql, tuple(params or ()))
        return [cast(dict[str, Any], row) for row in rows]


__all__ = ["PostgresDatabasePlugin"]
