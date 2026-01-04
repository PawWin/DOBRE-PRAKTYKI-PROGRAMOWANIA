from __future__ import annotations

import json
import logging
import time

import pika
import requests

from ..rabbitmq import connection_parameters
from ..settings import get_settings


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("result_sender")


def _send_result(payload: dict) -> None:
    settings = get_settings()
    endpoint = f"{settings.service_a_base_url}/results"
    response = requests.post(endpoint, json=payload, timeout=10)
    response.raise_for_status()


def _handle_message(
    ch: pika.channel.Channel,
    method: pika.spec.Basic.Deliver,
    _: pika.spec.BasicProperties,
    body: bytes,
) -> None:
    try:
        payload = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError:
        logger.error("Invalid message payload, dropping message")
        ch.basic_ack(delivery_tag=method.delivery_tag)
        return
    try:
        _send_result(payload)
        ch.basic_ack(delivery_tag=method.delivery_tag)
        logger.info("Delivered result %s to service A", payload.get("id"))
    except Exception as exc:  # pragma: no cover - depends on external services
        logger.warning("Failed to deliver result %s: %s", payload.get("id"), exc)
        time.sleep(5)
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)


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
    channel.queue_declare(queue=settings.result_queue, durable=True)
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=settings.result_queue, on_message_callback=_handle_message)
    logger.info("Waiting for results on queue %s", settings.result_queue)
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        logger.info("Stopping result sender...")
    finally:
        channel.stop_consuming()
        connection.close()


if __name__ == "__main__":
    main()
