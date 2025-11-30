from pydantic import BaseModel, Field


class MovieCreate(BaseModel):
    title: str
    genres: str


class MovieUpdate(BaseModel):
    title: str | None = None
    genres: str | None = None


class LinkCreate(BaseModel):
    movie_id: int
    imdb_id: str | None = None
    tmdb_id: str | None = None


class LinkUpdate(BaseModel):
    imdb_id: str | None = None
    tmdb_id: str | None = None


class RatingCreate(BaseModel):
    user_id: int
    movie_id: int
    rating: float
    timestamp: int


class RatingUpdate(BaseModel):
    user_id: int | None = None
    movie_id: int | None = None
    rating: float | None = None
    timestamp: int | None = None


class TagCreate(BaseModel):
    user_id: int
    movie_id: int
    tag: str
    timestamp: int


class TagUpdate(BaseModel):
    user_id: int | None = None
    movie_id: int | None = None
    tag: str | None = None
    timestamp: int | None = None


class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    roles: list[str] = Field(default_factory=lambda: ["ROLE_USER"])


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


def _dump(model: BaseModel, *, exclude_unset: bool = False) -> dict:
    """Compatibility helper for Pydantic v1/v2 model dumping."""
    if hasattr(model, "model_dump"):
        return model.model_dump(exclude_unset=exclude_unset)
    return model.dict(exclude_unset=exclude_unset)
