from __future__ import annotations

import json
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, HttpUrl

from app.rabbitmq import publish_task
from app.storage import result_path_for


router = APIRouter()


class AnalyzeRequest(BaseModel):
    url: HttpUrl


class AnalyzeResponse(BaseModel):
    id: str
    status: str
    check_url: str


class CheckResponse(BaseModel):
    id: str
    status: str
    count: Optional[int] = None
    image_path: Optional[str] = None
    error: Optional[str] = None


def _enqueue_job(url: str) -> AnalyzeResponse:
    job_id = str(uuid.uuid4())
    payload = {"id": job_id, "url": url}
    try:
        publish_task(payload)
    except Exception as exc:  # pragma: no cover - depends on external service
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Queue unavailable: {exc}",
        ) from exc
    return AnalyzeResponse(id=job_id, status="queued", check_url=f"/check?id={job_id}")


@router.post("/analyze_img", response_model=AnalyzeResponse)
def analyze_img_post(payload: AnalyzeRequest) -> AnalyzeResponse:
    return _enqueue_job(str(payload.url))


@router.get("/analyze_img", response_model=AnalyzeResponse)
def analyze_img_get(url: HttpUrl = Query(...)) -> AnalyzeResponse:
    return _enqueue_job(str(url))


@router.get("/check", response_model=CheckResponse)
def check(job_id: str = Query(..., alias="id")) -> CheckResponse:
    result_path = result_path_for(job_id)
    if not result_path.exists():
        return CheckResponse(id=job_id, status="pending")

    try:
        data = result_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read result: {exc}") from exc

    try:
        parsed = json.loads(data)
        result = CheckResponse(**parsed)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Invalid result data: {exc}") from exc

    return result
