from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

import pika
import requests

from app.rabbitmq import connection_parameters
from app.settings import get_settings
from app.storage import get_storage_dirs, result_path_for


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("consumer")


def detect_people_count(image_bytes: bytes) -> int:
    import cv2
    import numpy as np

    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to decode image bytes")

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    boxes, _weights = hog.detectMultiScale(gray, winStride=(8, 8))
    return int(len(boxes))


def _guess_extension(url: str, content_type: Optional[str]) -> str:
    if content_type:
        content_type = content_type.split(";")[0].strip().lower()
        if content_type == "image/jpeg":
            return ".jpg"
        if content_type == "image/png":
            return ".png"
        if content_type == "image/webp":
            return ".webp"
        if content_type == "image/bmp":
            return ".bmp"

    path = urlparse(url).path
    suffix = Path(path).suffix
    return suffix if suffix else ".jpg"


def _download_image(url: str) -> Tuple[bytes, str]:
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    ext = _guess_extension(url, response.headers.get("Content-Type"))
    return response.content, ext


def _write_result(job_id: str, data: Dict[str, Any]) -> None:
    result_path = result_path_for(job_id)
    result_path.write_text(json.dumps(data, ensure_ascii=True), encoding="utf-8")


def _handle_message(ch: pika.channel.Channel, method: pika.spec.Basic.Deliver, _: pika.spec.BasicProperties, body: bytes) -> None:
    try:
        payload = json.loads(body.decode("utf-8"))
        job_id = payload["id"]
        url = payload["url"]
        image_bytes, ext = _download_image(url)
        count = detect_people_count(image_bytes)
        images_dir, _ = get_storage_dirs()
        image_name = f"{job_id}-{count}{ext}"
        image_path = images_dir / image_name
        image_path.write_bytes(image_bytes)

        _write_result(
            job_id,
            {
                "id": job_id,
                "status": "done",
                "count": count,
                "image_path": str(image_path),
            },
        )
        ch.basic_ack(delivery_tag=method.delivery_tag)
        logger.info("Processed job %s (count=%s)", job_id, count)
    except Exception as exc:  # pragma: no cover - depends on external services
        job_id = None
        try:
            payload = json.loads(body.decode("utf-8"))
            job_id = payload.get("id")
        except Exception:
            pass

        if job_id:
            _write_result(
                job_id,
                {
                    "id": job_id,
                    "status": "failed",
                    "error": str(exc),
                },
            )
        ch.basic_ack(delivery_tag=method.delivery_tag)
        logger.exception("Failed to process message")


def main() -> None:
    settings = get_settings()
    connection = pika.BlockingConnection(connection_parameters())
    channel = connection.channel()
    channel.queue_declare(queue=settings.rabbitmq_queue, durable=True)
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=settings.rabbitmq_queue, on_message_callback=_handle_message)
    logger.info("Waiting for messages on queue %s", settings.rabbitmq_queue)
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        logger.info("Stopping consumer...")
    finally:
        channel.stop_consuming()
        connection.close()


if __name__ == "__main__":
    main()
