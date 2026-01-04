from __future__ import annotations

import json
import logging

import pika
import requests
import time

from ..rabbitmq import connection_parameters, publish_result_task
from ..settings import get_settings


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("analyze_worker")


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


def _download_image(url: str) -> bytes:
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    return response.content


def _handle_message(ch: pika.channel.Channel, method: pika.spec.Basic.Deliver, _: pika.spec.BasicProperties, body: bytes) -> None:
    try:
        payload = json.loads(body.decode("utf-8"))
        job_id = payload["id"]
        url = payload["url"]
        image_bytes = _download_image(url)
        count = detect_people_count(image_bytes)
        result_payload = {
            "id": job_id,
            "url": url,
            "status": "done",
            "count": count,
        }
        try:
            publish_result_task(result_payload)
        except Exception:
            logger.exception("Failed to publish result for %s", job_id)
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
            return
        ch.basic_ack(delivery_tag=method.delivery_tag)
        logger.info("Processed job %s (count=%s)", job_id, count)
    except Exception as exc:  # pragma: no cover - depends on external services
        job_id = None
        url = None
        try:
            payload = json.loads(body.decode("utf-8"))
            job_id = payload.get("id")
            url = payload.get("url")
        except Exception:
            pass

        if job_id:
            result_payload = {
                "id": job_id,
                "url": url,
                "status": "failed",
                "error": str(exc),
            }
            try:
                publish_result_task(result_payload)
            except Exception:
                logger.exception("Failed to publish result for %s", job_id)
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
                return
        ch.basic_ack(delivery_tag=method.delivery_tag)
        logger.exception("Failed to process message")


def _connect_with_retry() -> pika.BlockingConnection:
    while True:
        try:
            return pika.BlockingConnection(connection_parameters())
        except pika.exceptions.AMQPConnectionError:
            logger.warning("RabbitMQ unavailable, retrying in 5s...")
            time.sleep(5)


def main() -> None:
    settings = get_settings()
    connection = _connect_with_retry()
    channel = connection.channel()
    channel.queue_declare(queue=settings.image_queue, durable=True)
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=settings.image_queue, on_message_callback=_handle_message)
    logger.info("Waiting for messages on queue %s", settings.image_queue)
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        logger.info("Stopping consumer...")
    finally:
        channel.stop_consuming()
        connection.close()


if __name__ == "__main__":
    main()
