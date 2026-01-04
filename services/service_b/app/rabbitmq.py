from __future__ import annotations

import json

import pika

from .settings import get_settings


def connection_parameters() -> pika.ConnectionParameters:
    settings = get_settings()
    credentials = pika.PlainCredentials(settings.rabbitmq_user, settings.rabbitmq_pass)
    return pika.ConnectionParameters(
        host=settings.rabbitmq_host,
        port=settings.rabbitmq_port,
        credentials=credentials,
        heartbeat=30,
        blocked_connection_timeout=30,
    )


def _publish(payload: dict, queue_name: str) -> None:
    connection = pika.BlockingConnection(connection_parameters())
    channel = connection.channel()
    channel.queue_declare(queue=queue_name, durable=True)
    channel.basic_publish(
        exchange="",
        routing_key=queue_name,
        body=json.dumps(payload).encode("utf-8"),
        properties=pika.BasicProperties(delivery_mode=2),
    )
    connection.close()


def publish_image_task(payload: dict) -> None:
    settings = get_settings()
    _publish(payload, settings.image_queue)


def publish_result_task(payload: dict) -> None:
    settings = get_settings()
    _publish(payload, settings.result_queue)
