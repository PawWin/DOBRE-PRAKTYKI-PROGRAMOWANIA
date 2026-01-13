from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Ensure .env is loaded for local runs (api/worker)
load_dotenv()


class Settings(BaseSettings):
    """Application configuration loaded from environment variables with sane defaults."""

    rabbit_url: str = Field("amqp://guest:guest@rabbitmq:5672/", env="RABBIT_URL")
    queue_name: str = Field("alpr_jobs", env="QUEUE_NAME")
    rabbit_prefetch: int = Field(5, env="RABBIT_PREFETCH")

    sqlite_path: str = Field("/app/data/app.db", env="SQLITE_PATH")

    model_path: str = Field("./LP-detection.pt", env="MODEL_PATH")
    use_gpu: bool = Field(False, env="USE_GPU")

    http_timeout: float = Field(10.0, env="HTTP_TIMEOUT")
    max_image_size: int = Field(8 * 1024 * 1024, env="MAX_IMAGE_SIZE")  # bytes

    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")

    class Config:
        env_file = ".env"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
