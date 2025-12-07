from __future__ import annotations

import argparse
from typing import Dict, List

from queue_utils import (
    QUEUE_DB,
    STATUS_DONE,
    STATUS_IN_PROGRESS,
    STATUS_PENDING,
    create_jobs,
    utc_now,
)


def enqueue_jobs(
    count: int, description_template: str | None, status: str
) -> List[int]:
    template = description_template or "Telephone conversation #{id}"

    def build_row(job_id: int) -> Dict[str, str | int]:
        timestamp = utc_now()
        return {
            "status": status,
            "created_at": timestamp,
            "updated_at": timestamp,
            "description": template.format(id=job_id),
        }

    return create_jobs(count, build_row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adds new jobs to the SQLite-backed queue."
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
        f"Created {len(created_ids)} job(s) in {QUEUE_DB.name}: "
        + ", ".join(str(job_id) for job_id in created_ids)
    )


if __name__ == "__main__":
    main()
