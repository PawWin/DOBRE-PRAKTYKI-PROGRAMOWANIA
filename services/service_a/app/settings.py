from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Settings:
    storage_dir: Path


def get_settings() -> Settings:
    storage_root = os.getenv("STORAGE_DIR")
    if storage_root:
        storage_dir = Path(storage_root)
    else:
        storage_dir = Path(__file__).resolve().parent / "storage"

    return Settings(storage_dir=storage_dir)
