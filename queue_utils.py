
from __future__ import annotations

import csv
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

BASE_DIR = Path(__file__).resolve().parent
QUEUE_FILE = BASE_DIR / "jobs_queue.csv"
LOCK_FILE = BASE_DIR / "jobs_queue.lock"

STATUS_PENDING = "pending"
STATUS_IN_PROGRESS = "in_progress"
STATUS_DONE = "done"

QUEUE_HEADERS = ["id", "status", "created_at", "updated_at", "description"]


class FileLock:
    """Lightweight cross-platform lock built on a temporary file."""

    def __init__(self, lock_path: Path, retry_delay: float = 0.05):
        self.lock_path = Path(lock_path)
        self.retry_delay = retry_delay
        self._fd: int | None = None

    def acquire(self) -> None:
        while True:
            try:
                self._fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                return
            except FileExistsError:
                time.sleep(self.retry_delay)

    def release(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        if self.lock_path.exists():
            try:
                self.lock_path.unlink()
            except FileNotFoundError:
                pass

    def __enter__(self) -> "FileLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.release()


def ensure_queue_file() -> None:
    if QUEUE_FILE.exists():
        return
    with open(QUEUE_FILE, "w", newline="", encoding="utf-8") as queue_file:
        writer = csv.DictWriter(queue_file, fieldnames=QUEUE_HEADERS)
        writer.writeheader()


def _normalize_job(row: Dict[str, str]) -> Dict[str, str | int]:
    row["id"] = int(row["id"])
    return row


def read_jobs() -> List[Dict[str, str | int]]:
    ensure_queue_file()
    with open(QUEUE_FILE, newline="", encoding="utf-8") as queue_file:
        reader = csv.DictReader(queue_file)
        return [_normalize_job(row) for row in reader]


def write_jobs(jobs: List[Dict[str, str | int]]) -> None:
    with open(QUEUE_FILE, "w", newline="", encoding="utf-8") as queue_file:
        writer = csv.DictWriter(queue_file, fieldnames=QUEUE_HEADERS)
        writer.writeheader()
        for job in jobs:
            serialized = {**job, "id": str(job["id"])}
            writer.writerow(serialized)


def utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")
