from fastapi import FastAPI

from .endpoints.results import router as results_router


app = FastAPI(title="Service A - Results API", version="1.0.0")
app.include_router(results_router)
