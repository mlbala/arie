from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from arie_db_sqlite.plugin import SQLiteDatabasePlugin


@pytest.mark.asyncio
async def test_database_contract_paper_isolation_and_lease(tmp_path: Path) -> None:
    db = SQLiteDatabasePlugin(tmp_path / "phase1b.sqlite")

    await db.upsert_paper_and_chunks(
        {
            "paper_id": "paper-1",
            "title": "Paper One",
            "source": "arxiv",
            "pdf_url": "https://example.com/p1.pdf",
        },
        [
            {"chunk_id": "chunk-1", "paper_id": "paper-1", "text": "alpha", "order": 0},
            {"chunk_id": "chunk-2", "paper_id": "paper-1", "text": "beta", "order": 1},
        ],
    )
    await db.upsert_paper_and_chunks(
        {
            "paper_id": "paper-2",
            "title": "Paper Two",
            "source": "arxiv",
            "pdf_url": "https://example.com/p2.pdf",
        },
        [{"chunk_id": "chunk-3", "paper_id": "paper-2", "text": "gamma", "order": 0}],
    )

    rows = await db.get_paper_chunks("paper-1")
    assert len(rows) == 2
    assert all(row["paper_id"] == "paper-1" for row in rows)

    await db.execute(
        None,
        (
            "INSERT INTO jobs (job_id, job_kind, payload_json, status, created_at, "
            "updated_at) VALUES (?, ?, ?, 'queued', ?, ?)"
        ),
        ("job-1", "ingest", "{}", "2025-01-01T00:00:00Z", "2025-01-01T00:00:00Z"),
    )
    lease = await db.claim_job()
    assert lease is not None
    assert lease["job_id"] == "job-1"
    assert lease["status"] == "leased"


@pytest.mark.asyncio
async def test_database_contract_atomic_rollback_on_partial_failure(tmp_path: Path) -> None:
    db = SQLiteDatabasePlugin(tmp_path / "phase1b_rollback.sqlite")

    def failing_transaction(conn: sqlite3.Connection) -> None:
        conn.execute(
            "INSERT INTO papers (paper_id, title, source, pdf_url) VALUES (?, ?, ?, ?)",
            ("paper-rollback", "Rollback", "arxiv", "https://example.com/rollback.pdf"),
        )
        conn.execute(
            (
                'INSERT INTO chunks (chunk_id, paper_id, section, text, "order", '
                "token_count) VALUES (?, ?, ?, ?, ?, ?)"
            ),
            ("chunk-rollback", "paper-rollback", "body", "oops", 0, 4),
        )
        raise RuntimeError("simulate partial failure")

    with pytest.raises(RuntimeError, match="simulate partial failure"):
        await db.atomic(failing_transaction)

    count = await db.fetch(
        None,
        "SELECT COUNT(*) AS c FROM papers WHERE paper_id = ?",
        ("paper-rollback",),
    )
    assert count[0]["c"] == 0
