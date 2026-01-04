from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, HttpUrl

from ..rabbitmq import publish_image_task
from ..settings import get_settings


router = APIRouter()


class AnalyzeRequest(BaseModel):
    url: HttpUrl


class AnalyzeResponse(BaseModel):
    id: str
    status: str
    check_url: str

def _enqueue_job(url: str) -> AnalyzeResponse:
    settings = get_settings()
    job_id = str(uuid.uuid4())
    payload = {"id": job_id, "url": url}
    try:
        publish_image_task(payload)
    except Exception as exc:  # pragma: no cover - depends on external service
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Queue unavailable: {exc}",
        ) from exc
    return AnalyzeResponse(
        id=job_id,
        status="queued",
        check_url=f"{settings.service_a_public_url}/results/{job_id}",
    )


@router.post("/analyze_img", response_model=AnalyzeResponse)
def analyze_img_post(payload: AnalyzeRequest) -> AnalyzeResponse:
    return _enqueue_job(str(payload.url))


@router.get("/analyze_img", response_model=AnalyzeResponse)
def analyze_img_get(url: HttpUrl = Query(...)) -> AnalyzeResponse:
    return _enqueue_job(str(url))
