from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

import aio_pika
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl

from src.config import get_settings
from src.db import get_job, init_db, insert_job, job_to_dict, update_job
from src.image_fetcher import HttpImageFetcher
from src.pipeline import PlateRecognitionPipeline

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="ALPR API", version="1.0.0")


class AnalyzeRequest(BaseModel):
    image_url: HttpUrl


class EnqueueRequest(BaseModel):
    image_url: HttpUrl


def _serialize_bbox(bbox: tuple[int, int, int, int] | None) -> list[int] | None:
    if bbox is None:
        return None
    return list(map(int, bbox))


@app.on_event("startup")
async def startup_event() -> None:
    settings = get_settings()
    init_db(settings.sqlite_path)

    app.state.settings = settings
    app.state.fetcher = HttpImageFetcher(
        timeout=settings.http_timeout, max_size=settings.max_image_size
    )
    app.state.pipeline = PlateRecognitionPipeline(
        detector_model=settings.model_path,
        use_gpu=settings.use_gpu,
        imgsz=1056,
    )

    app.state.rabbit_conn = await aio_pika.connect_robust(settings.rabbit_url)
    app.state.rabbit_channel = await app.state.rabbit_conn.channel()
    await app.state.rabbit_channel.set_qos(prefetch_count=settings.rabbit_prefetch)
    await app.state.rabbit_channel.declare_queue(settings.queue_name, durable=True)

    logger.info("Startup complete - queue declared, DB ready, pipeline loaded.")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    fetcher: HttpImageFetcher | None = getattr(app.state, "fetcher", None)
    if fetcher:
        await fetcher.aclose()
    rabbit_conn: aio_pika.RobustConnection | None = getattr(app.state, "rabbit_conn", None)
    if rabbit_conn:
        await rabbit_conn.close()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(req: AnalyzeRequest) -> JSONResponse:
    settings = app.state.settings
    fetcher: HttpImageFetcher = app.state.fetcher
    pipeline: PlateRecognitionPipeline = app.state.pipeline

    try:
        image = await fetcher.fetch(str(req.image_url))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to fetch image")
        raise HTTPException(status_code=400, detail=f"Cannot fetch image: {exc}") from exc

    start = time.perf_counter()
    try:
        result = await run_in_threadpool(pipeline.process_image, image)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Pipeline failed")
        raise HTTPException(status_code=500, detail="Processing failed") from exc

    duration_ms = (time.perf_counter() - start) * 1000

    if result is None:
        return JSONResponse(
            {
                "status": "no_plate",
                "duration_ms": duration_ms,
                "image_url": str(req.image_url),
            },
            status_code=200,
        )

    response: dict[str, Any] = {
        "status": "ok",
        "duration_ms": duration_ms,
        "image_url": str(req.image_url),
        "text": result.get("text"),
        "bbox": _serialize_bbox(result.get("bbox")),
    }
    return JSONResponse(response)


@app.post("/enqueue")
async def enqueue(req: EnqueueRequest) -> dict[str, Any]:
    settings = app.state.settings
    job_id = str(uuid.uuid4())

    insert_job(settings.sqlite_path, job_id=job_id, image_url=str(req.image_url))

    message = aio_pika.Message(
        body=json.dumps({"job_id": job_id, "image_url": str(req.image_url)}).encode(),
        delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
    )
    await app.state.rabbit_channel.default_exchange.publish(
        message, routing_key=settings.queue_name
    )

    return {"job_id": job_id, "status": "queued"}


@app.get("/job/{job_id}")
async def get_job_status(job_id: str) -> dict[str, Any]:
    settings = app.state.settings
    record = get_job(settings.sqlite_path, job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found")
    data = job_to_dict(record)
    data["bbox"] = _serialize_bbox(record.bbox)
    return data
