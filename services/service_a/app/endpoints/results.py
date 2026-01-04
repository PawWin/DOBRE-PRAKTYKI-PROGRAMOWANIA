from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, HttpUrl

from ..storage import result_path_for


router = APIRouter()


class ResultIn(BaseModel):
    id: str
    url: HttpUrl
    status: str
    count: Optional[int] = None
    error: Optional[str] = None


class ResultOut(ResultIn):
    received_at: str


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.post("/results", response_model=ResultOut, status_code=status.HTTP_201_CREATED)
def create_result(payload: ResultIn) -> ResultOut:
    received_at = datetime.now(timezone.utc).isoformat()
    data = jsonable_encoder(payload)
    data["received_at"] = received_at

    result_path = result_path_for(payload.id)
    try:
        result_path.write_text(json.dumps(data, ensure_ascii=True), encoding="utf-8")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to store result: {exc}") from exc

    return ResultOut(**data)


@router.get("/results/{job_id}", response_model=ResultOut)
def get_result(job_id: str) -> ResultOut:
    result_path = result_path_for(job_id)
    if not result_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Result not found")

    try:
        data = json.loads(result_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read result: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"Invalid result data: {exc}") from exc

    return ResultOut(**data)
