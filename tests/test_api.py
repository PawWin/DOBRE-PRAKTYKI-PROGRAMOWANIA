from itertools import count
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.api import app
from app.db import get_db
from app.models import Base, Link, Movie, Rating, Tag


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
    def _override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture()
def movie_factory(db_session):
    sequence = count(1)

    def _create_movie(**kwargs):
        idx = next(sequence)
        movie = Movie(
            title=kwargs.get("title", f"Movie {idx}"),
            genres=kwargs.get("genres", "Drama"),
        )
        db_session.add(movie)
        db_session.commit()
        db_session.refresh(movie)
        return movie

    return _create_movie


@pytest.fixture()
def link_factory(db_session, movie_factory):
    sequence = count(1)

    def _create_link(*, movie=None, **kwargs):
        idx = next(sequence)
        related_movie = movie or movie_factory()
        link = Link(
            movie_id=kwargs.get("movie_id", related_movie.id),
            imdb_id=kwargs.get("imdb_id", f"tt{idx:07d}"),
            tmdb_id=kwargs.get("tmdb_id", f"{100000 + idx}"),
        )
        db_session.add(link)
        db_session.commit()
        db_session.refresh(link)
        return link

    return _create_link


@pytest.fixture()
def rating_factory(db_session, movie_factory):
    sequence = count(1)

    def _create_rating(*, movie=None, **kwargs):
        idx = next(sequence)
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
def tag_factory(db_session, movie_factory):
    sequence = count(1)

    def _create_tag(*, movie=None, **kwargs):
        idx = next(sequence)
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


def test_root_hello_world(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"hello": "world"}


def test_get_movies_returns_seeded_entries(client, movie_factory):
    movie_a = movie_factory()
    movie_b = movie_factory()

    response = client.get("/movies")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    returned_ids = {item["id"] for item in data}
    assert returned_ids == {movie_a.id, movie_b.id}


def test_create_movie_persists_and_returns_new_movie(client, db_session):
    payload = {"title": "New Movie", "genres": "Comedy"}

    response = client.post("/movies", json=payload)
    assert response.status_code == 201
    body = response.json()
    assert body["title"] == payload["title"]
    assert body["genres"] == payload["genres"]
    assert db_session.query(Movie).count() == 1


def test_get_movie_returns_specific_movie(client, movie_factory):
    movie = movie_factory(title="Original", genres="Drama|Action")

    response = client.get(f"/movies/{movie.id}")
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == movie.id
    assert body["title"] == "Original"
    assert body["genres"] == "Drama|Action"


def test_get_movie_missing_returns_404(client):
    response = client.get("/movies/9999")
    assert response.status_code == 404
    assert response.json()["detail"] == "Movie not found"


def test_update_movie_modifies_existing_record(client, movie_factory, db_session):
    movie = movie_factory(title="Old Title", genres="Action")
    payload = {"title": "Updated Title"}

    response = client.put(f"/movies/{movie.id}", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["title"] == "Updated Title"
    assert body["genres"] == "Action"
    db_session.refresh(movie)
    assert movie.title == "Updated Title"


def test_delete_movie_removes_record(client, movie_factory, db_session):
    movie = movie_factory()

    response = client.delete(f"/movies/{movie.id}")
    assert response.status_code == 204
    assert db_session.query(Movie).count() == 0


def test_get_links_returns_seeded_entries(client, link_factory, movie_factory):
    movie_a = movie_factory()
    movie_b = movie_factory()
    link_a = link_factory(movie=movie_a, imdb_id="tt0000001", tmdb_id="1001")
    link_b = link_factory(movie=movie_b, imdb_id="tt0000002", tmdb_id="1002")

    response = client.get("/links")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    movie_ids = {item["movie_id"] for item in data}
    assert movie_ids == {link_a.movie_id, link_b.movie_id}


def test_create_link_persists_new_entry(client, movie_factory, db_session):
    movie = movie_factory()
    payload = {"movie_id": movie.id, "imdb_id": "tt7654321", "tmdb_id": "98765"}

    response = client.post("/links", json=payload)
    assert response.status_code == 201
    body = response.json()
    assert body["movie_id"] == movie.id
    assert body["tmdb_id"] == payload["tmdb_id"]
    stored = db_session.query(Link).filter_by(movie_id=movie.id).one()
    assert stored.imdb_id == payload["imdb_id"]


def test_get_link_returns_single_entry(client, link_factory):
    link = link_factory()

    response = client.get(f"/links/{link.movie_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["movie_id"] == link.movie_id
    assert body["imdb_id"] == link.imdb_id


def test_update_link_overwrites_fields(client, link_factory, db_session):
    link = link_factory(imdb_id="tt1111111", tmdb_id="1111")
    payload = {"tmdb_id": "2222"}

    response = client.put(f"/links/{link.movie_id}", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["tmdb_id"] == "2222"
    db_session.refresh(link)
    assert link.tmdb_id == "2222"


def test_delete_link_removes_entry(client, link_factory, db_session):
    link = link_factory()

    response = client.delete(f"/links/{link.movie_id}")
    assert response.status_code == 204
    assert db_session.query(Link).count() == 0


def test_get_ratings_returns_seeded_entries(client, rating_factory):
    rating_factory(rating=4.5)
    rating_factory(rating=3.0)

    response = client.get("/ratings")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert {item["rating"] for item in data} == {4.5, 3.0}


def test_create_rating_persists_new_entry(client, movie_factory, db_session):
    movie = movie_factory()
    payload = {"user_id": 77, "movie_id": movie.id, "rating": 4.5, "timestamp": 123456789}

    response = client.post("/ratings", json=payload)
    assert response.status_code == 201
    body = response.json()
    assert body["user_id"] == 77
    assert body["rating"] == pytest.approx(4.5)
    stored = db_session.query(Rating).filter_by(user_id=77).one()
    assert stored.timestamp == payload["timestamp"]


def test_get_rating_returns_single_entry(client, rating_factory):
    rating = rating_factory(rating=2.5)

    response = client.get(f"/ratings/{rating.id}")
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == rating.id
    assert body["rating"] == pytest.approx(2.5)


def test_get_rating_missing_returns_404(client):
    response = client.get("/ratings/9999")
    assert response.status_code == 404
    assert response.json()["detail"] == "Rating not found"


def test_update_rating_modifies_existing_resource(client, rating_factory, db_session):
    rating = rating_factory(rating=1.5)
    new_timestamp = rating.timestamp + 10
    payload = {"rating": 4.0, "timestamp": new_timestamp}

    response = client.put(f"/ratings/{rating.id}", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["rating"] == pytest.approx(4.0)
    assert body["timestamp"] == new_timestamp
    db_session.refresh(rating)
    assert rating.rating == pytest.approx(4.0)
    assert rating.timestamp == new_timestamp


def test_delete_rating_removes_existing_entry(client, rating_factory, db_session):
    rating = rating_factory()

    response = client.delete(f"/ratings/{rating.id}")
    assert response.status_code == 204
    assert db_session.query(Rating).count() == 0


def test_get_tags_returns_seeded_entries(client, tag_factory):
    tag_factory(tag="Sci-Fi")
    tag_factory(tag="Drama")

    response = client.get("/tags")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert {"Sci-Fi", "Drama"} == {item["tag"] for item in data}


def test_create_tag_persists_entry(client, movie_factory, db_session):
    movie = movie_factory()
    payload = {"user_id": 42, "movie_id": movie.id, "tag": "Favorite", "timestamp": 987654321}

    response = client.post("/tags", json=payload)
    assert response.status_code == 201
    body = response.json()
    assert body["tag"] == "Favorite"
    assert body["movie_id"] == movie.id
    stored = db_session.query(Tag).filter_by(user_id=42).one()
    assert stored.timestamp == payload["timestamp"]


def test_get_tag_returns_single_entry(client, tag_factory):
    tag = tag_factory(tag="Top Pick")

    response = client.get(f"/tags/{tag.id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == tag.id
    assert data["tag"] == "Top Pick"


def test_update_tag_modifies_existing_entry(client, tag_factory, db_session):
    tag = tag_factory(tag="Old Value")
    payload = {"tag": "Updated Value"}

    response = client.put(f"/tags/{tag.id}", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["tag"] == "Updated Value"
    db_session.refresh(tag)
    assert tag.tag == "Updated Value"


def test_delete_tag_removes_entry(client, tag_factory, db_session):
    tag = tag_factory()

    response = client.delete(f"/tags/{tag.id}")
    assert response.status_code == 204
    assert db_session.query(Tag).count() == 0
