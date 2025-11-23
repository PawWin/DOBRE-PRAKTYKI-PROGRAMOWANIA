import sys
from pathlib import Path

from fastapi import Depends, FastAPI
from sqlalchemy.orm import Session

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from app.db import get_db, init_db, serialize_model  # noqa: E402
from app.models import Link, Movie, Rating, Tag  # noqa: E402

app = FastAPI()
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
    import sys

    # Ensure project root is on sys.path so uvicorn reload can import "app.api:app".
    root_dir = BASE_DIR.parent
    if str(root_dir) not in sys.path:
        sys.path.append(str(root_dir))

    import uvicorn

    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)
