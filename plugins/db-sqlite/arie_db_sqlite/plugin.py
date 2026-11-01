from __future__ import annotations

import asyncio
import json
import queue
import sqlite3
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

from arie_core.plugin_api import plugin


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _utc_plus(seconds: float) -> str:
    return (datetime.now(UTC) + timedelta(seconds=seconds)).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )


@plugin(slot="database", name="sqlite", api=">=1,<2")
class SQLiteDatabasePlugin:
    """SQLite-backed database plugin with WAL durability and serialized writes.

    The writer path is serialized behind a single thread to preserve atomic journal
    semantics while read-only access uses a lightweight pool of independent SQLite
    connections. This keeps the writer fast and deterministic, and it also enforces
    paper-level post-filtering before returning rows to preserve Invariant I1.
    """

    _MIGRATION_PATH = (
        Path(__file__).resolve().parents[3]
        / "core"
        / "arie_core"
        / "persistence"
        / "migrations"
        / "0001_arie_v6_initial.sql"
    )

    def __init__(
        self,
        db_path: str | Path = "data/arie.sqlite3",
        *,
        read_pool_size: int = 4,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._read_pool_size = max(1, read_pool_size)
        self._read_index = 0
        self._read_connections: list[sqlite3.Connection] = [
            self._connect_read() for _ in range(self._read_pool_size)
        ]
        self._writer_lock = threading.Lock()
        self._write_queue: queue.Queue[
            tuple[Callable[[sqlite3.Connection], object], dict[str, object], threading.Event]
        ] = queue.Queue()
        self._writer_conn = self._connect_writer()
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name="arie-sqlite-writer",
            daemon=True,
        )
        self._writer_thread.start()
        self._initialize_database()

    def _connect_read(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            str(self.db_path),
            timeout=0.25,
            isolation_level=None,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        return connection

    def _connect_writer(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            isolation_level=None,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_database(self) -> None:
        with self._writer_lock:
            self._writer_conn.execute("PRAGMA journal_mode = WAL;")
            self._writer_conn.execute("PRAGMA synchronous = NORMAL;")
            self._writer_conn.execute("PRAGMA busy_timeout = 5000;")
            self._writer_conn.execute("PRAGMA foreign_keys = ON;")
            self._writer_conn.execute("PRAGMA temp_store = MEMORY;")
            self._writer_conn.execute("PRAGMA mmap_size = 268435456;")
            script = self._MIGRATION_PATH.read_text(encoding="utf-8")
            self._writer_conn.executescript(script)

    def _writer_loop(self) -> None:
        while True:
            item = self._write_queue.get()
            callback, state, ready = item
            try:
                state["result"] = callback(self._writer_conn)
            except BaseException as exc:  # pragma: no cover - transactional safeguard
                state["error"] = exc
            finally:
                ready.set()

    def _submit_write(self, callback: Callable[[sqlite3.Connection], object]) -> object:
        state: dict[str, object] = {}
        ready = threading.Event()
        self._write_queue.put((callback, state, ready))
        ready.wait()
        error = state.get("error")
        if error is not None:
            raise cast(BaseException, error)
        return state.get("result")

    @staticmethod
    def _coerce_int(value: object, default: int = 0) -> int:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return default
        return default

    def _read_one(
        self,
        query: str,
        params: Sequence[object] = (),
    ) -> list[dict[str, object]]:
        connection = self._read_connections[self._read_index]
        self._read_index = (self._read_index + 1) % self._read_pool_size
        cursor = connection.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    async def execute(
        self,
        ctx: object | None = None,
        query: str | None = None,
        params: Sequence[object] = (),
    ) -> int:
        if query is None:
            raise ValueError("execute requires a SQL query")

        def _run(conn: sqlite3.Connection) -> int:
            cursor = conn.execute(query, params)
            rowcount = cursor.rowcount
            return self._coerce_int(rowcount)

        result = await asyncio.to_thread(self._submit_write, _run)
        return self._coerce_int(result)

    async def fetch(
        self,
        ctx: object | None = None,
        query: str | None = None,
        params: Sequence[object] = (),
    ) -> list[dict[str, object]]:
        if query is None:
            raise ValueError("fetch requires a SQL query")
        return await asyncio.to_thread(self._read_one, query, params)

    async def atomic(
        self,
        callback: Callable[[sqlite3.Connection], object],
        *,
        ctx: object | None = None,
    ) -> object:
        def _run(conn: sqlite3.Connection) -> object:
            conn.execute("BEGIN IMMEDIATE")
            try:
                result = callback(conn)
                conn.execute("COMMIT")
                return result
            except Exception:
                conn.execute("ROLLBACK")
                raise

        return await asyncio.to_thread(self._submit_write, _run)

    async def upsert_paper_and_chunks(
        self,
        paper: Mapping[str, object] | None = None,
        chunks: Sequence[Mapping[str, object]] | None = None,
        *,
        ctx: object | None = None,
    ) -> str:
        if paper is None:
            raise ValueError("paper payload is required")

        paper_id = str(paper.get("paper_id") or paper.get("id") or "")
        if not paper_id:
            raise ValueError("paper_id is required to upsert a paper")

        paper_payload = dict(paper)
        paper_payload["paper_id"] = paper_id
        paper_payload["updated_at"] = _utc_now()

        def _run(conn: sqlite3.Connection) -> str:
            conn.execute(
                """
                INSERT INTO papers (
                    paper_id,
                    title,
                    source,
                    pdf_url,
                    abstract,
                    metadata_json,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(paper_id) DO UPDATE SET
                    title = excluded.title,
                    source = excluded.source,
                    pdf_url = excluded.pdf_url,
                    abstract = excluded.abstract,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                (
                    paper_id,
                    str(paper_payload.get("title") or ""),
                    str(paper_payload.get("source") or ""),
                    paper_payload.get("pdf_url"),
                    paper_payload.get("abstract"),
                    json.dumps(paper_payload.get("metadata") or {}, sort_keys=True),
                    paper_payload.get("created_at") or _utc_now(),
                    paper_payload["updated_at"],
                ),
            )

            for index, chunk in enumerate(chunks or []):
                chunk_payload = dict(chunk)
                chunk_id = str(chunk_payload.get("chunk_id") or f"{paper_id}:{index}")
                section = str(chunk_payload.get("section") or "body")
                text = str(chunk_payload.get("text") or "")
                order_value = chunk_payload.get("order")
                token_value = chunk_payload.get("token_count")
                order = self._coerce_int(order_value, index)
                token_count = self._coerce_int(
                    token_value,
                    max(0, len(text) // 4),
                )
                conn.execute(
                    """
                    INSERT INTO chunks (
                        chunk_id,
                        paper_id,
                        section,
                        text,
                        "order",
                        token_count,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(chunk_id) DO UPDATE SET
                        paper_id = excluded.paper_id,
                        section = excluded.section,
                        text = excluded.text,
                        "order" = excluded."order",
                        token_count = excluded.token_count
                    """,
                    (
                        chunk_id,
                        paper_id,
                        section,
                        text,
                        order,
                        token_count,
                        chunk_payload.get("created_at") or _utc_now(),
                    ),
                )
            return paper_id

        return str(await asyncio.to_thread(self._submit_write, _run))

    async def get_paper_chunks(
        self,
        *args: object,
        **kwargs: object,
    ) -> list[dict[str, object]]:
        ctx = kwargs.get("ctx")
        paper_id_value = kwargs.get("paper_id")
        query_value = kwargs.get("query")
        limit_value = kwargs.get("limit")

        paper_id: str | None = str(paper_id_value) if paper_id_value is not None else None
        query: str | None = str(query_value) if query_value is not None else None
        limit: int | None = self._coerce_int(limit_value) if limit_value is not None else None

        if args:
            if isinstance(args[0], str):
                paper_id = str(args[0])
            elif isinstance(args[0], Mapping):
                mapping = cast(Mapping[str, object], args[0])
                raw_paper_id = mapping.get("paper_id") or mapping.get("id")
                paper_id = str(raw_paper_id) if raw_paper_id is not None else None
        if args and len(args) > 1 and isinstance(args[1], str):
            query = args[1]

        if not paper_id:
            raise ValueError("paper_id is required to fetch chunks")

        sql = 'SELECT * FROM chunks WHERE paper_id = ?'
        params: list[object] = [paper_id]
        if query:
            sql += ' AND text LIKE ?'
            params.append(f"%{query}%")
        sql += ' ORDER BY "order" ASC'
        if limit is not None:
            sql += ' LIMIT ?'
            params.append(limit)

        rows = await self.fetch(ctx, sql, tuple(params))
        return [row for row in rows if str(row.get("paper_id")) == str(paper_id)]

    async def claim_job(
        self,
        *,
        ctx: object | None = None,
        lease_seconds: float = 60.0,
        claim_token: str | None = None,
    ) -> dict[str, object] | None:
        token = claim_token or uuid.uuid4().hex
        lease_until = _utc_plus(lease_seconds)
        sql = """
            UPDATE jobs
            SET status = 'leased',
                lease_until = ?,
                claim_token = ?,
                attempts = attempts + 1,
                updated_at = ?
            WHERE job_id = (
                SELECT job_id FROM jobs
                WHERE status IN ('queued', 'leased')
                  AND (lease_until IS NULL OR lease_until <= ?)
                ORDER BY created_at ASC
                LIMIT 1
            )
            RETURNING *
        """
        rows = await self.fetch(ctx, sql, (lease_until, token, _utc_now(), _utc_now()))
        if not rows:
            return None
        return dict(rows[0])

    async def claim_next_job(
        self,
        *,
        ctx: object | None = None,
        lease_seconds: float = 60.0,
        claim_token: str | None = None,
    ) -> dict[str, object] | None:
        return await self.claim_job(ctx=ctx, lease_seconds=lease_seconds, claim_token=claim_token)

    async def lease_job(
        self,
        *,
        ctx: object | None = None,
        lease_seconds: float = 60.0,
        claim_token: str | None = None,
    ) -> dict[str, object] | None:
        return await self.claim_job(ctx=ctx, lease_seconds=lease_seconds, claim_token=claim_token)

    async def enqueue_outbox(
        self,
        topic: str,
        payload: Mapping[str, object] | Sequence[object] | str | None,
        *,
        ctx: object | None = None,
    ) -> int:
        payload_json = json.dumps(payload, sort_keys=True) if payload is not None else "{}"
        sql = (
            "INSERT INTO outbox (topic, payload_json, status, created_at) "
            "VALUES (?, ?, 'pending', ?)"
        )
        rowcount = await self.execute(ctx, sql, (topic, payload_json, _utc_now()))
        return int(rowcount)

    async def claim_outbox(
        self,
        *,
        ctx: object | None = None,
        limit: int = 10,
        lease_seconds: float = 60.0,
        lease_token: str | None = None,
    ) -> list[dict[str, object]]:
        token = lease_token or uuid.uuid4().hex
        lease_until = _utc_plus(lease_seconds)
        sql = """
            UPDATE outbox
            SET status = 'claimed',
                leased_until = ?,
                lease_token = ?
            WHERE id IN (
                SELECT id FROM outbox
                WHERE status = 'pending'
                  AND (leased_until IS NULL OR leased_until <= ?)
                ORDER BY created_at ASC
                LIMIT ?
            )
            RETURNING *
        """
        return await self.fetch(ctx, sql, (lease_until, token, _utc_now(), int(limit)))

    async def complete_outbox(
        self,
        outbox_id: int,
        *,
        ctx: object | None = None,
        lease_token: str | None = None,
    ) -> int:
        sql = (
            "UPDATE outbox SET status = 'sent', leased_until = NULL, "
            "lease_token = NULL WHERE id = ?"
        )
        params: list[object] = [outbox_id]
        if lease_token:
            sql += " AND lease_token = ?"
            params.append(lease_token)
        return await self.execute(ctx, sql, tuple(params))

    async def search_bm25(
        self,
        query: str,
        *,
        ctx: object | None = None,
        paper_id: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, object]]:
        normalized = query.strip()
        if not normalized:
            return []
        sql = (
            "SELECT rowid, chunk_id, paper_id, section, text, bm25(chunks_fts) AS score "
            "FROM chunks_fts WHERE chunks_fts MATCH ?"
        )
        params: list[object] = [normalized]
        if paper_id:
            sql += " AND paper_id = ?"
            params.append(paper_id)
        sql += " ORDER BY bm25(chunks_fts) ASC LIMIT ?"
        params.append(int(limit))
        rows = await self.fetch(ctx, sql, tuple(params))
        return [row for row in rows if paper_id is None or str(row.get("paper_id")) == paper_id]


__all__ = ["SQLiteDatabasePlugin"]
