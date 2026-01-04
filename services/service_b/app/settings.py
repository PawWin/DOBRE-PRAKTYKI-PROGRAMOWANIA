from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    rabbitmq_host: str
    rabbitmq_port: int
    rabbitmq_user: str
    rabbitmq_pass: str
    image_queue: str
    result_queue: str
    service_a_base_url: str
    service_a_public_url: str


def get_settings() -> Settings:
    service_a_base_url = os.getenv("SERVICE_A_URL", "http://service-a:8000")
    service_a_public_url = os.getenv("SERVICE_A_PUBLIC_URL", service_a_base_url)

    return Settings(
        rabbitmq_host=os.getenv("RABBITMQ_HOST", "localhost"),
        rabbitmq_port=int(os.getenv("RABBITMQ_PORT", "5672")),
        rabbitmq_user=os.getenv("RABBITMQ_USER", "guest"),
        rabbitmq_pass=os.getenv("RABBITMQ_PASS", "guest"),
        image_queue=os.getenv("RABBITMQ_IMAGE_QUEUE", "image_tasks"),
        result_queue=os.getenv("RABBITMQ_RESULT_QUEUE", "result_tasks"),
        service_a_base_url=service_a_base_url,
        service_a_public_url=service_a_public_url,
    )
