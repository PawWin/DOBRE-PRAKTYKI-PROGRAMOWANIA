from __future__ import annotations

from pathlib import Path
from typing import Tuple

from app.settings import get_settings


def get_storage_dirs() -> Tuple[Path, Path]:
    settings = get_settings()
    images_dir = settings.storage_dir / "images"
    results_dir = settings.storage_dir / "results"
    images_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    return images_dir, results_dir


def result_path_for(job_id: str) -> Path:
    _, results_dir = get_storage_dirs()
    return results_dir / f"{job_id}.json"
