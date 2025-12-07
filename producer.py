from __future__ import annotations
import argparse
from typing import List, Dict

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


def _build_job(
    job_id: int, description: str, status: str = STATUS_PENDING
) -> Dict[str, str | int]:
    timestamp = utc_now()
    return {
        "id": job_id,
        "status": status,
        "created_at": timestamp,
        "updated_at": timestamp,
        "description": description,
    }


def enqueue_jobs(
    count: int, description_template: str | None, status: str
) -> List[int]:
    ensure_queue_file()
    with FileLock(LOCK_FILE):
        jobs = read_jobs()
        last_id = max((job["id"] for job in jobs), default=0)
        created_ids = []
        for offset in range(count):
            job_id = last_id + offset + 1
            desc = (
                description_template.format(id=job_id)
                if description_template
                else f"Telephone conversation #{job_id}"
            )
            jobs.append(_build_job(job_id, desc, status=status))
            created_ids.append(job_id)
        write_jobs(jobs)
    return created_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adds new jobs to the file-based queue."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of jobs to enqueue (default: 1).",
    )
    parser.add_argument(
        "--description",
        type=str,
        default=None,
        help="Custom description template, accepts {id} placeholder.",
    )
    parser.add_argument(
        "--status",
        type=str,
        choices=[STATUS_PENDING, STATUS_IN_PROGRESS, STATUS_DONE],
        default=STATUS_PENDING,
        help="Status assigned to new jobs (default: pending).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise SystemExit("Count must be a positive integer.")

    created_ids = enqueue_jobs(args.count, args.description, args.status)
    print(
        f"Created {len(created_ids)} job(s) in {QUEUE_FILE.name}: "
        + ", ".join(str(job_id) for job_id in created_ids)
    )


if __name__ == "__main__":
    main()
