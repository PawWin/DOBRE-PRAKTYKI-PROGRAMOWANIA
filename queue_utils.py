from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent
QUEUE_DB = BASE_DIR / "jobs_queue.sqlite3"

STATUS_PENDING = "pending"
STATUS_IN_PROGRESS = "in_progress"
STATUS_DONE = "done"

RowDict = Dict[str, str | int]

_SCHEMA_INITIALIZED = False


def utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def ensure_database() -> None:
    global _SCHEMA_INITIALIZED
    if _SCHEMA_INITIALIZED and QUEUE_DB.exists():
        return
    with sqlite3.connect(QUEUE_DB) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                description TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)"
        )
        conn.commit()
    _SCHEMA_INITIALIZED = True


@contextmanager
def db_connection():
    ensure_database()
    conn = sqlite3.connect(QUEUE_DB)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def create_jobs(count: int, build_row: Callable[[int], RowDict]) -> List[int]:
    ensure_database()
    if count <= 0:
        return []
    with db_connection() as conn:
        conn.isolation_level = None
        conn.execute("BEGIN IMMEDIATE")
        try:
            last_id_row = conn.execute(
                "SELECT COALESCE(MAX(id), 0) FROM jobs"
            ).fetchone()
            last_id = int(last_id_row[0]) if last_id_row else 0
            created_ids: List[int] = []
            for offset in range(count):
                job_id = last_id + offset + 1
                row = build_row(job_id)
                conn.execute(
                    """
                    INSERT INTO jobs (id, status, created_at, updated_at, description)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        job_id,
                        row["status"],
                        row["created_at"],
                        row["updated_at"],
                        row["description"],
                    ),
                )
                created_ids.append(job_id)
            conn.execute("COMMIT")
            return created_ids
        except Exception:
            conn.execute("ROLLBACK")
            raise


def reserve_pending_job() -> Optional[RowDict]:
    ensure_database()
    with db_connection() as conn:
        conn.isolation_level = None
        conn.execute("BEGIN IMMEDIATE")
        try:
            row = conn.execute(
                """
                SELECT id, status, created_at, updated_at, description
                FROM jobs
                WHERE status = ?
                ORDER BY id
                LIMIT 1
                """,
                (STATUS_PENDING,),
            ).fetchone()
            if row is None:
                conn.execute("ROLLBACK")
                return None
            job_id = int(row["id"])
            now = utc_now()
            updated = conn.execute(
                """
                UPDATE jobs
                SET status = ?, updated_at = ?
                WHERE id = ? AND status = ?
                """,
                (STATUS_IN_PROGRESS, now, job_id, STATUS_PENDING),
            )
            if updated.rowcount == 0:
                conn.execute("ROLLBACK")
                return None
            conn.execute("COMMIT")
            job = dict(row)
            job["id"] = job_id
            job["status"] = STATUS_IN_PROGRESS
            job["updated_at"] = now
            return job
        except Exception:
            conn.execute("ROLLBACK")
            raise


def mark_job_done(job_id: int) -> bool:
    ensure_database()
    with db_connection() as conn:
        now = utc_now()
        cur = conn.execute(
            "UPDATE jobs SET status = ?, updated_at = ? WHERE id = ?",
            (STATUS_DONE, now, job_id),
        )
        conn.commit()
        return cur.rowcount > 0
