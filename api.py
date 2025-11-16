from pathlib import Path
import csv
from typing import List

from fastapi import FastAPI

app = FastAPI()


BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "database"


class Movie:
    def __init__(self, movieId: str, title: str, genres: str):
        self.id = movieId
        self.title = title
        self.genres = genres


class Link:
    def __init__(self, movieId: str, imdbId: str, tmdbId: str):
        self.movieId = movieId
        self.imdbId = imdbId
        self.tmdbId = tmdbId


class Rating:
    def __init__(self, userId: str, movieId: str, rating: str, timestamp: str):
        self.userId = userId
        self.movieId = movieId
        self.rating = rating
        self.timestamp = timestamp


class Tag:
    def __init__(self, userId: str, movieId: str, tag: str, timestamp: str):
        self.userId = userId
        self.movieId = movieId
        self.tag = tag
        self.timestamp = timestamp


def read_csv(file_path: Path, cls: type) -> List[dict]:
    """Load a CSV file and return a list of serialized model instances."""
    items = []
    with file_path.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            items.append(cls(**row).__dict__)
    return items


movies_data = read_csv(DB_DIR / "movies.csv", Movie)
links_data = read_csv(DB_DIR / "links.csv", Link)
ratings_data = read_csv(DB_DIR / "ratings.csv", Rating)
tags_data = read_csv(DB_DIR / "tags.csv", Tag)


@app.get("/")
def hello_world():
    return {"hello": "world"}


@app.get("/movies")
def get_movies():
    return movies_data


@app.get("/links")
def get_links():
    return links_data


@app.get("/ratings")
def get_ratings():
    return ratings_data


@app.get("/tags")
def get_tags():
    return tags_data


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
