from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Settings:
    rabbitmq_host: str
    rabbitmq_port: int
    rabbitmq_user: str
    rabbitmq_pass: str
    rabbitmq_queue: str
    storage_dir: Path


def get_settings() -> Settings:
    storage_root = os.getenv("STORAGE_DIR")
    if storage_root:
        storage_dir = Path(storage_root)
    else:
        storage_dir = Path(__file__).resolve().parent / "storage"

    return Settings(
        rabbitmq_host=os.getenv("RABBITMQ_HOST", "localhost"),
        rabbitmq_port=int(os.getenv("RABBITMQ_PORT", "5672")),
        rabbitmq_user=os.getenv("RABBITMQ_USER", "guest"),
        rabbitmq_pass=os.getenv("RABBITMQ_PASS", "guest"),
        rabbitmq_queue=os.getenv("RABBITMQ_QUEUE", "image_tasks"),
        storage_dir=storage_dir,
    )
