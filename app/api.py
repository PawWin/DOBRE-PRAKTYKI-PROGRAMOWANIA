import sys
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Response, status
from sqlalchemy.orm import Session


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from app.db import get_db, init_db, serialize_model
from app.schemas import (LinkCreate, LinkUpdate, MovieCreate, MovieUpdate,
                         RatingCreate, RatingUpdate, TagCreate, TagUpdate, _dump)
from app.models import Link, Movie, Rating, Tag
from app.endpoints.auth import get_current_user, router as auth_router
from app.endpoints.user import router as user_router

app = FastAPI()
init_db()
app.include_router(auth_router)
app.include_router(user_router)

def _get_movie_or_404(movie_id: int, db: Session) -> Movie:
    movie = db.query(Movie).filter(Movie.id == movie_id).first()
    if not movie:
        raise HTTPException(status_code=404, detail="Movie not found")
    return movie


def _get_link_or_404(movie_id: int, db: Session) -> Link:
    link = db.query(Link).filter(Link.movie_id == movie_id).first()
    if not link:
        raise HTTPException(status_code=404, detail="Link not found")
    return link


def _get_rating_or_404(rating_id: int, db: Session) -> Rating:
    rating = db.query(Rating).filter(Rating.id == rating_id).first()
    if not rating:
        raise HTTPException(status_code=404, detail="Rating not found")
    return rating


def _get_tag_or_404(tag_id: int, db: Session) -> Tag:
    tag = db.query(Tag).filter(Tag.id == tag_id).first()
    if not tag:
        raise HTTPException(status_code=404, detail="Tag not found")
    return tag


def _ensure_movie_exists(db: Session, movie_id: int) -> None:
    db_movie = db.query(Movie).filter(Movie.id == movie_id).first()
    if not db_movie:
        raise HTTPException(status_code=400, detail="Movie does not exist")


@app.get("/", dependencies=[Depends(get_current_user)])
def hello_world():
    return {"hello": "world"}


@app.get("/movies", dependencies=[Depends(get_current_user)])
def get_movies(db: Session = Depends(get_db)):
    movies = db.query(Movie).all()
    return [serialize_model(movie) for movie in movies]


@app.post("/movies", status_code=status.HTTP_201_CREATED, dependencies=[Depends(get_current_user)])
def create_movie(movie_in: MovieCreate, db: Session = Depends(get_db)):
    movie = Movie(**_dump(movie_in))
    db.add(movie)
    db.commit()
    db.refresh(movie)
    return serialize_model(movie)


@app.get("/movies/{movie_id}", dependencies=[Depends(get_current_user)])
def get_movie(movie_id: int, db: Session = Depends(get_db)):
    movie = _get_movie_or_404(movie_id, db)
    return serialize_model(movie)


@app.put("/movies/{movie_id}", dependencies=[Depends(get_current_user)])
def update_movie(movie_id: int, movie_in: MovieUpdate, db: Session = Depends(get_db)):
    movie = _get_movie_or_404(movie_id, db)
    for field, value in _dump(movie_in, exclude_unset=True).items():
        setattr(movie, field, value)
    db.commit()
    db.refresh(movie)
    return serialize_model(movie)


@app.delete("/movies/{movie_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(get_current_user)])
def delete_movie(movie_id: int, db: Session = Depends(get_db)):
    movie = _get_movie_or_404(movie_id, db)
    db.delete(movie)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.get("/links", dependencies=[Depends(get_current_user)])
def get_links(db: Session = Depends(get_db)):
    links = db.query(Link).all()
    return [serialize_model(link) for link in links]


@app.post("/links", status_code=status.HTTP_201_CREATED, dependencies=[Depends(get_current_user)])
def create_link(link_in: LinkCreate, db: Session = Depends(get_db)):
    _ensure_movie_exists(db, link_in.movie_id)
    existing = db.query(Link).filter(Link.movie_id == link_in.movie_id).first()
    if existing:
        raise HTTPException(status_code=400, detail="Link already exists for this movie")
    link = Link(**_dump(link_in))
    db.add(link)
    db.commit()
    db.refresh(link)
    return serialize_model(link)


@app.get("/links/{movie_id}", dependencies=[Depends(get_current_user)])
def get_link(movie_id: int, db: Session = Depends(get_db)):
    link = _get_link_or_404(movie_id, db)
    return serialize_model(link)


@app.put("/links/{movie_id}", dependencies=[Depends(get_current_user)])
def update_link(movie_id: int, link_in: LinkUpdate, db: Session = Depends(get_db)):
    link = _get_link_or_404(movie_id, db)
    for field, value in _dump(link_in, exclude_unset=True).items():
        setattr(link, field, value)
    db.commit()
    db.refresh(link)
    return serialize_model(link)


@app.delete("/links/{movie_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(get_current_user)])
def delete_link(movie_id: int, db: Session = Depends(get_db)):
    link = _get_link_or_404(movie_id, db)
    db.delete(link)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.get("/ratings", dependencies=[Depends(get_current_user)])
def get_ratings(db: Session = Depends(get_db)):
    ratings = db.query(Rating).all()
    return [serialize_model(rating) for rating in ratings]


@app.post("/ratings", status_code=status.HTTP_201_CREATED, dependencies=[Depends(get_current_user)])
def create_rating(rating_in: RatingCreate, db: Session = Depends(get_db)):
    _ensure_movie_exists(db, rating_in.movie_id)
    rating = Rating(**_dump(rating_in))
    db.add(rating)
    db.commit()
    db.refresh(rating)
    return serialize_model(rating)


@app.get("/ratings/{rating_id}", dependencies=[Depends(get_current_user)])
def get_rating(rating_id: int, db: Session = Depends(get_db)):
    rating = _get_rating_or_404(rating_id, db)
    return serialize_model(rating)


@app.put("/ratings/{rating_id}", dependencies=[Depends(get_current_user)])
def update_rating(rating_id: int, rating_in: RatingUpdate, db: Session = Depends(get_db)):
    rating = _get_rating_or_404(rating_id, db)
    update_data = _dump(rating_in, exclude_unset=True)
    if "movie_id" in update_data:
        _ensure_movie_exists(db, update_data["movie_id"])
    for field, value in update_data.items():
        setattr(rating, field, value)
    db.commit()
    db.refresh(rating)
    return serialize_model(rating)


@app.delete("/ratings/{rating_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(get_current_user)])
def delete_rating(rating_id: int, db: Session = Depends(get_db)):
    rating = _get_rating_or_404(rating_id, db)
    db.delete(rating)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.get("/tags", dependencies=[Depends(get_current_user)])
def get_tags(db: Session = Depends(get_db)):
    tags = db.query(Tag).all()
    return [serialize_model(tag) for tag in tags]


@app.post("/tags", status_code=status.HTTP_201_CREATED, dependencies=[Depends(get_current_user)])
def create_tag(tag_in: TagCreate, db: Session = Depends(get_db)):
    _ensure_movie_exists(db, tag_in.movie_id)
    tag = Tag(**_dump(tag_in))
    db.add(tag)
    db.commit()
    db.refresh(tag)
    return serialize_model(tag)


@app.get("/tags/{tag_id}", dependencies=[Depends(get_current_user)])
def get_tag(tag_id: int, db: Session = Depends(get_db)):
    tag = _get_tag_or_404(tag_id, db)
    return serialize_model(tag) 


@app.put("/tags/{tag_id}", dependencies=[Depends(get_current_user)])
def update_tag(tag_id: int, tag_in: TagUpdate, db: Session = Depends(get_db)):
    tag = _get_tag_or_404(tag_id, db)
    update_data = _dump(tag_in, exclude_unset=True)
    if "movie_id" in update_data:
        _ensure_movie_exists(db, update_data["movie_id"])
    for field, value in update_data.items():
        setattr(tag, field, value)
    db.commit()
    db.refresh(tag)
    return serialize_model(tag)


@app.delete("/tags/{tag_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(get_current_user)])
def delete_tag(tag_id: int, db: Session = Depends(get_db)):
    tag = _get_tag_or_404(tag_id, db)
    db.delete(tag)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


if __name__ == "__main__":
    import sys

    # Ensure project root is on sys.path so uvicorn reload can import "app.api:app".
    root_dir = BASE_DIR.parent
    if str(root_dir) not in sys.path:
        sys.path.append(str(root_dir))

    import uvicorn

    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)
