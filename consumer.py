
from __future__ import annotations

import argparse
import time
from typing import Dict

from queue_utils import (
    FileLock,
    LOCK_FILE,
    QUEUE_FILE,
    STATUS_DONE,
    STATUS_IN_PROGRESS,
    STATUS_PENDING,
    ensure_queue_file,
    read_jobs,
    write_jobs,
    utc_now,
)


def reserve_job() -> Dict[str, str | int] | None:
    ensure_queue_file()
    with FileLock(LOCK_FILE):
        jobs = read_jobs()
        for job in jobs:
            if job["status"] == STATUS_PENDING:
                job["status"] = STATUS_IN_PROGRESS
                job["updated_at"] = utc_now()
                write_jobs(jobs)
                return job
    return None


def mark_done(job_id: int) -> bool:
    with FileLock(LOCK_FILE):
        jobs = read_jobs()
        for job in jobs:
            if job["id"] == job_id:
                job["status"] = STATUS_DONE
                job["updated_at"] = utc_now()
                write_jobs(jobs)
                return True
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continuously consumes jobs from the file-based queue."
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Delay between file checks in seconds (default: 5).",
    )
    parser.add_argument(
        "--work-duration",
        type=float,
        default=30.0,
        help="Simulated job execution time in seconds (default: 30).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.poll_interval <= 0:
        raise SystemExit("poll-interval must be positive.")
    if args.work_duration <= 0:
        raise SystemExit("work-duration must be positive.")

    print(
        f"Consumer started. Queue file: {QUEUE_FILE}. "
        f"Polling every {args.poll_interval}s."
    )
    try:
        while True:
            job = reserve_job()
            if not job:
                time.sleep(args.poll_interval)
                continue
            job_id = job["id"]
            description = job["description"]
            print(
                f"Job {job_id} ({description}) in progress for {args.work_duration}s."
            )
            time.sleep(args.work_duration)
            if mark_done(job_id):
                print(f"Job {job_id} completed and marked as done.")
            else:
                print(f"Job {job_id} missing while finishing, skipping.")
    except KeyboardInterrupt:
        print("Consumer interrupted, shutting down.")


if __name__ == "__main__":
    main()
