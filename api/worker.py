from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

import aio_pika

from src.config import get_settings
from src.db import get_job, init_db, insert_job, update_job
from src.image_fetcher import HttpImageFetcher
from src.pipeline import PlateRecognitionPipeline

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


async def process_one(
    message: aio_pika.IncomingMessage,
    pipeline: PlateRecognitionPipeline,
    fetcher: HttpImageFetcher,
    settings,
) -> None:
    async with message.process():
        payload = json.loads(message.body)
        job_id = payload["job_id"]
        image_url = payload["image_url"]

        logger.info("Processing job %s", job_id)
        # Ensure job exists (idempotent)
        if get_job(settings.sqlite_path, job_id) is None:
            insert_job(settings.sqlite_path, job_id=job_id, image_url=image_url)

        update_job(settings.sqlite_path, job_id, status="running")

        start = time.perf_counter()
        try:
            image = await fetcher.fetch(image_url)
            result = await asyncio.to_thread(pipeline.process_image, image)
            duration_ms = (time.perf_counter() - start) * 1000

            if result is None:
                update_job(
                    settings.sqlite_path,
                    job_id,
                    status="done",
                    result_text=None,
                    bbox=None,
                    duration_ms=duration_ms,
                )
            else:
                update_job(
                    settings.sqlite_path,
                    job_id,
                    status="done",
                    result_text=result.get("text"),
                    bbox=result.get("bbox"),
                    duration_ms=duration_ms,
                )
            logger.info("Job %s done in %.0f ms", job_id, duration_ms)
        except Exception as exc:  # noqa: BLE001
            duration_ms = (time.perf_counter() - start) * 1000
            update_job(
                settings.sqlite_path,
                job_id,
                status="error",
                error=str(exc),
                duration_ms=duration_ms,
            )
            logger.exception("Job %s failed", job_id)


async def main() -> None:
    settings = get_settings()
    init_db(settings.sqlite_path)

    fetcher = HttpImageFetcher(
        timeout=settings.http_timeout, max_size=settings.max_image_size
    )
    pipeline = PlateRecognitionPipeline(
        detector_model=settings.model_path,
        use_gpu=settings.use_gpu,
        imgsz=1056,
    )

    connection = await aio_pika.connect_robust(settings.rabbit_url)
    channel = await connection.channel()
    await channel.set_qos(prefetch_count=settings.rabbit_prefetch)
    queue = await channel.declare_queue(settings.queue_name, durable=True)

    logger.info("Worker started. Waiting for messages on %s", settings.queue_name)
    try:
        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                await process_one(message, pipeline, fetcher, settings)
    finally:
        await fetcher.aclose()
        await connection.close()


if __name__ == "__main__":
    asyncio.run(main())
