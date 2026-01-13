from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class JobRecord:
    job_id: str
    image_url: str
    status: str
    created_at: str
    updated_at: str
    finished_at: str | None = None
    result_text: str | None = None
    bbox: tuple[int, int, int, int] | None = None
    duration_ms: float | None = None
    error: str | None = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                image_url TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                finished_at TEXT,
                result_text TEXT,
                bbox TEXT,
                duration_ms REAL,
                error TEXT
            );
            """
        )
        conn.commit()


def insert_job(db_path: str, job_id: str, image_url: str) -> None:
    now = _now_iso()
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO jobs (
                job_id, image_url, status, created_at, updated_at
            ) VALUES (?, ?, 'queued', ?, ?)
            """,
            (job_id, image_url, now, now),
        )
        conn.commit()


def update_job(
    db_path: str,
    job_id: str,
    status: str | None = None,
    result_text: str | None = None,
    bbox: tuple[int, int, int, int] | None = None,
    duration_ms: float | None = None,
    error: str | None = None,
) -> None:
    updates: dict[str, Any] = {"updated_at": _now_iso()}
    if status is not None:
        updates["status"] = status
    if result_text is not None:
        updates["result_text"] = result_text
    if bbox is not None:
        updates["bbox"] = json.dumps(bbox)
    if duration_ms is not None:
        updates["duration_ms"] = duration_ms
    if error is not None:
        updates["error"] = error
    if status in {"done", "error"}:
        updates["finished_at"] = _now_iso()

    columns = ", ".join([f"{k} = ?" for k in updates])
    values = list(updates.values())
    values.append(job_id)

    with _connect(db_path) as conn:
        conn.execute(f"UPDATE jobs SET {columns} WHERE job_id = ?", values)
        conn.commit()


def get_job(db_path: str, job_id: str) -> JobRecord | None:
    with _connect(db_path) as conn:
        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        bbox = json.loads(row["bbox"]) if row["bbox"] else None
        return JobRecord(
            job_id=row["job_id"],
            image_url=row["image_url"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            finished_at=row["finished_at"],
            result_text=row["result_text"],
            bbox=tuple(bbox) if bbox else None,
            duration_ms=row["duration_ms"],
            error=row["error"],
        )


def job_to_dict(record: JobRecord) -> dict[str, Any]:
    data = asdict(record)
    if record.bbox is None:
        data["bbox"] = None
    return data
