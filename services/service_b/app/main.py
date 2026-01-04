from fastapi import FastAPI

from .endpoints.analyze import router as analyze_router


app = FastAPI(title="Service B - Analyzer API", version="1.0.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


app.include_router(analyze_router)
