from __future__ import annotations

import argparse
import time

from queue_utils import (
    QUEUE_DB,
    mark_job_done,
    reserve_pending_job,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continuously consumes jobs from the SQLite-backed queue."
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
        f"Consumer started. Queue database: {QUEUE_DB}. "
        f"Polling every {args.poll_interval}s."
    )
    try:
        while True:
            job = reserve_pending_job()
            if not job:
                time.sleep(args.poll_interval)
                continue
            job_id = job["id"]
            description = job["description"]
            print(
                f"Job {job_id} ({description}) in progress for {args.work_duration}s."
            )
            time.sleep(args.work_duration)
            if mark_job_done(job_id):
                print(f"Job {job_id} completed and marked as done.")
            else:
                print(f"Job {job_id} missing while finishing, skipping.")
    except KeyboardInterrupt:
        print("Consumer interrupted, shutting down.")


if __name__ == "__main__":
    main()
