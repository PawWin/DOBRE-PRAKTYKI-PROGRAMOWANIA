from fastapi import FastAPI

from app.endpoints.analyze import router as analyze_router


app = FastAPI(title="Image Analyzer API", version="0.1.0")
app.include_router(analyze_router)
