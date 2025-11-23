from pathlib import Path
import csv
from typing import Callable, Iterable

from fastapi import Depends, FastAPI
from sqlalchemy import Column, Float, ForeignKey, Integer, String, create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

app = FastAPI()


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "database"
DB_DIR = BASE_DIR / "database"
DB_FILE = DB_DIR / "movies.sqlite"
DATABASE_URL = f"sqlite:///{DB_FILE}"

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class Movie(Base):
    __tablename__ = "movies"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    genres = Column(String, nullable=False)


class Link(Base):
    __tablename__ = "links"

    movie_id = Column(Integer, ForeignKey("movies.id"), primary_key=True)
    imdb_id = Column(String, nullable=True)
    tmdb_id = Column(String, nullable=True)


class Rating(Base):
    __tablename__ = "ratings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=False, index=True)
    movie_id = Column(Integer, ForeignKey("movies.id"), nullable=False, index=True)
    rating = Column(Float, nullable=False)
    timestamp = Column(Integer, nullable=False)


class Tag(Base):
    __tablename__ = "tags"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=False, index=True)
    movie_id = Column(Integer, ForeignKey("movies.id"), nullable=False, index=True)
    tag = Column(String, nullable=False)
    timestamp = Column(Integer, nullable=False)


def get_db() -> Iterable[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _to_int(value: str | None) -> int | None:
    return int(value) if value not in (None, "") else None


def _to_float(value: str | None) -> float | None:
    return float(value) if value not in (None, "") else None


def seed_table(
    session: Session,
    model: type[Base],
    csv_path: Path,
    serializer: Callable[[dict[str, str]], dict],
) -> None:
    """Populate a table from CSV if it's empty."""
    if session.query(model).first():
        return

    with csv_path.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        items = [model(**serializer(row)) for row in reader]

    session.add_all(items)
    session.commit()


def init_db() -> None:
    DB_DIR.mkdir(exist_ok=True)
    Base.metadata.create_all(bind=engine)

    with SessionLocal() as session:
        seed_table(
            session,
            Movie,
            DATA_DIR / "movies.csv",
            lambda row: {
                "id": _to_int(row["movieId"]),
                "title": row["title"],
                "genres": row["genres"],
            },
        )
        seed_table(
            session,
            Link,
            DATA_DIR / "links.csv",
            lambda row: {
                "movie_id": _to_int(row["movieId"]),
                "imdb_id": row.get("imdbId") or None,
                "tmdb_id": row.get("tmdbId") or None,
            },
        )
        seed_table(
            session,
            Rating,
            DATA_DIR / "ratings.csv",
            lambda row: {
                "user_id": _to_int(row["userId"]),
                "movie_id": _to_int(row["movieId"]),
                "rating": _to_float(row["rating"]),
                "timestamp": _to_int(row["timestamp"]),
            },
        )
        seed_table(
            session,
            Tag,
            DATA_DIR / "tags.csv",
            lambda row: {
                "user_id": _to_int(row["userId"]),
                "movie_id": _to_int(row["movieId"]),
                "tag": row.get("tag") or "",
                "timestamp": _to_int(row["timestamp"]),
            },
        )


def serialize_model(instance: Base) -> dict:
    return {column.name: getattr(instance, column.name) for column in instance.__table__.columns}


init_db()


@app.get("/")
def hello_world():
    return {"hello": "world"}


@app.get("/movies")
def get_movies(db: Session = Depends(get_db)):
    movies = db.query(Movie).all()
    return [serialize_model(movie) for movie in movies]


@app.get("/links")
def get_links(db: Session = Depends(get_db)):
    links = db.query(Link).all()
    return [serialize_model(link) for link in links]


@app.get("/ratings")
def get_ratings(db: Session = Depends(get_db)):
    ratings = db.query(Rating).all()
    return [serialize_model(rating) for rating in ratings]


@app.get("/tags")
def get_tags(db: Session = Depends(get_db)):
    tags = db.query(Tag).all()
    return [serialize_model(tag) for tag in tags]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)
