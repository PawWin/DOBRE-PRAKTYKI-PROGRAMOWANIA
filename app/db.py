from pathlib import Path
import csv
from typing import Callable, Iterable
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.models import Base, Link, Movie, Rating, Tag, User
from app.security import format_roles, hash_password

BASE_DIR = Path(__file__).resolve().parent
# CSV files and SQLite file both live in app/database
DATA_DIR = BASE_DIR / "database"
DB_DIR = BASE_DIR / "database"
DB_FILE = DB_DIR / "database.sqlite"
DATABASE_URL = f"sqlite:///{DB_FILE}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


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
        _ensure_admin_user(session)


def serialize_model(instance: Base) -> dict:
    return {column.name: getattr(instance, column.name) for column in instance.__table__.columns}


def _ensure_admin_user(session: Session) -> None:
    if session.query(User).filter(User.username == "admin").first():
        return
    admin = User(
        username="admin",
        email="admin@example.com",
        hashed_password=hash_password("admin"),
        roles=format_roles(["ROLE_ADMIN", "ROLE_USER"]),
    )
    session.add(admin)
    session.commit()


