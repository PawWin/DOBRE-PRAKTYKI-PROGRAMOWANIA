from __future__ import annotations

from pathlib import Path

from .settings import get_settings


def get_storage_dir() -> Path:
    settings = get_settings()
    storage_dir = settings.storage_dir
    storage_dir.mkdir(parents=True, exist_ok=True)
    return storage_dir


def result_path_for(job_id: str) -> Path:
    storage_dir = get_storage_dir()
    return storage_dir / f"{job_id}.json"
