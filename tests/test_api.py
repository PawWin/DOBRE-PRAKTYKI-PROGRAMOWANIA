from itertools import count
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from app.api import app
from app.db import get_db
from app.models import Base, Movie, Link, Rating, Tag

engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture()
def db_session():
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
    Base.metadata.drop_all(bind=engine)

@pytest.fixture()
def client(db_session):
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()

@pytest.fixture()
def movie_factory(db_session):
    movie_id_counter = count(1)

    def _create_movie(**kwargs):
        movie_id = next(movie_id_counter)
        movie = Movie(id=movie_id, title=kwargs.get("title", f"Movie {movie_id}"), genres=kwargs.get("genres", "Drama"))
        db_session.add(movie)
        db_session.commit()
        db_session.refresh(movie)
        return movie

    return _create_movie

@pytest.fixture()
def link_factory(db_session):
    link_id_counter = count(1)

    def _create_link(*,movie=None, **kwargs):
        link_id = next(link_id_counter)
        related_movie = movie or movie_factory()
        link = Link(
            movie_id=kwargs.get("movie_id", related_movie.id),
            imdb_id=kwargs.get("imdb_id", f"tt{link_id:07d}"),
            tmdb_id=kwargs.get("tmdb_id", f"{100000 + link_id}"),
        )
        db_session.add(link)
        db_session.commit()
        db_session.refresh(link)
        return link

    return _create_link

@pytest.fixture()
def rating_factory(db_session):
    rating_id_counter = count(1)

    def _create_rating(*, movie=None, **kwargs):
        idx = next(rating_id_counter)
        related_movie = movie or movie_factory()
        rating = Rating(
            user_id=kwargs.get("user_id", idx),
            movie_id=kwargs.get("movie_id", related_movie.id),
            rating=kwargs.get("rating", 4.0),
            timestamp=kwargs.get("timestamp", 100000 + idx),
        )
        db_session.add(rating)
        db_session.commit()
        db_session.refresh(rating)
        return rating

    return _create_rating

@pytest.fixture()
def tag_factory(db_session):
    tag_id_counter = count(1)

    def _create_tag(*, movie=None, **kwargs):
        idx = next(tag_id_counter)
        related_movie = movie or movie_factory()
        tag = Tag(
            user_id=kwargs.get("user_id", idx),
            movie_id=kwargs.get("movie_id", related_movie.id),
            tag=kwargs.get("tag", f"Tag {idx}"),
            timestamp=kwargs.get("timestamp", 200000 + idx),
        )
        db_session.add(tag)
        db_session.commit()
        db_session.refresh(tag)
        return tag

    return _create_tag

