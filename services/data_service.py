"""Load, validate, and prepare the public MovieLens small dataset."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd


DATASET_NAME = "MovieLens latest-small"
DATASET_VERSION = "Generated 2018-09-26; activity through 2018-09-24"
DATASET_URL = "https://grouplens.org/datasets/movielens/latest/"
DATASET_README_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html"
MIN_RATINGS_FOR_WEIGHTED_SCORE = 100
GENRE_LABELS = {
    name.casefold(): name
    for name in (
        "Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary",
        "Drama", "Fantasy", "Film-Noir", "Horror", "IMAX", "Musical", "Mystery",
        "Romance", "Sci-Fi", "Thriller", "War", "Western",
    )
}


class DatasetError(ValueError):
    """Raised when source files are missing or do not match the expected schema."""


def _read_csv(path: Path, required_columns: set[str], **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise DatasetError(f"Required dataset file is missing: {path.name}")
    frame = pd.read_csv(path, **kwargs)
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        raise DatasetError(f"{path.name} is missing columns: {', '.join(missing)}")
    return frame


def load_dataset(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return cleaned movies, ratings, tags, and IMDb-link frames."""
    movies = _read_csv(
        data_dir / "movies.csv", {"movieId", "title", "genres"},
        dtype={"title": "string", "genres": "string"},
    )
    ratings = _read_csv(
        data_dir / "ratings.csv", {"userId", "movieId", "rating", "timestamp"},
    )
    tags = _read_csv(
        data_dir / "tags.csv", {"userId", "movieId", "tag", "timestamp"},
        dtype={"tag": "string"},
    )
    links = _read_csv(
        data_dir / "links.csv", {"movieId", "imdbId", "tmdbId"},
        dtype={"imdbId": "string", "tmdbId": "string"},
    )

    movies["movieId"] = pd.to_numeric(movies["movieId"], errors="coerce")
    movies["movieId"] = movies["movieId"].where(movies["movieId"].mod(1).eq(0))
    movies["title"] = movies["title"].fillna("").str.strip()
    movies["genres"] = movies["genres"].fillna("").str.strip()
    year_match = movies["title"].str.extract(r"\((\d{4})\)\s*$", expand=False)
    movies["year"] = pd.to_numeric(year_match, errors="coerce")
    movies["year"] = movies["year"].where(movies["year"].between(1874, date.today().year + 5)).astype("Int64")
    movies["title"] = movies["title"].str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True).str.strip()
    movies["genre_list"] = movies["genres"].map(lambda value: list(dict.fromkeys(
        GENRE_LABELS.get(genre.strip().casefold(), genre.strip())
        for genre in value.split("|")
        if genre.strip() and genre.strip().casefold() != "(no genres listed)"
    )))
    movies = movies.dropna(subset=["movieId"])
    movies = movies[movies["title"].ne("")].drop_duplicates("movieId", keep="first")
    movies["movieId"] = movies["movieId"].astype("int64")

    ratings["userId"] = pd.to_numeric(ratings["userId"], errors="coerce")
    ratings["movieId"] = pd.to_numeric(ratings["movieId"], errors="coerce")
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    ratings["timestamp"] = pd.to_numeric(ratings["timestamp"], errors="coerce")
    ratings = ratings.dropna(subset=["userId", "movieId", "rating", "timestamp"])
    ratings = ratings[ratings["userId"].mod(1).eq(0) & ratings["movieId"].mod(1).eq(0)]
    ratings = ratings[ratings["rating"].between(0.5, 5.0)]
    ratings = ratings[ratings["movieId"].isin(movies["movieId"])]
    ratings = ratings.drop_duplicates(["userId", "movieId"], keep="last").copy()
    ratings["movieId"] = ratings["movieId"].astype("int64")
    ratings["rated_at"] = pd.to_datetime(ratings["timestamp"], unit="s", errors="coerce", utc=True)
    ratings = ratings.dropna(subset=["rated_at"])

    tags["userId"] = pd.to_numeric(tags["userId"], errors="coerce")
    tags["movieId"] = pd.to_numeric(tags["movieId"], errors="coerce")
    tags["timestamp"] = pd.to_numeric(tags["timestamp"], errors="coerce")
    tags["tag"] = tags["tag"].fillna("").str.strip()
    tags = tags.dropna(subset=["userId", "movieId", "timestamp"])
    tags = tags[tags["userId"].mod(1).eq(0) & tags["movieId"].mod(1).eq(0)]
    tags = tags[tags["tag"].ne("") & tags["movieId"].isin(movies["movieId"])]
    tags = tags.drop_duplicates(["userId", "movieId", "tag", "timestamp"]).copy()
    tags["movieId"] = tags["movieId"].astype("int64")
    tags["tagged_at"] = pd.to_datetime(tags["timestamp"], unit="s", errors="coerce", utc=True)
    tags = tags.dropna(subset=["tagged_at"])

    links["movieId"] = pd.to_numeric(links["movieId"], errors="coerce")
    links["movieId"] = links["movieId"].where(links["movieId"].mod(1).eq(0))
    links = links.dropna(subset=["movieId"]).drop_duplicates("movieId", keep="first")
    links["movieId"] = links["movieId"].astype("int64")
    links["imdbId"] = links["imdbId"].replace({"": pd.NA, "nan": pd.NA})

    return movies.reset_index(drop=True), ratings.reset_index(drop=True), tags.reset_index(drop=True), links.reset_index(drop=True)


def add_movie_statistics(movies: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """Add observed rating aggregates and the clearly labeled Bayesian score."""
    global_mean = float(ratings["rating"].mean()) if not ratings.empty else 0.0
    aggregates = ratings.groupby("movieId")["rating"].agg(avg_rating="mean", rating_count="count")
    result = movies.merge(aggregates, on="movieId", how="left")
    result["rating_count"] = result["rating_count"].fillna(0).astype("int64")
    result["avg_rating"] = result["avg_rating"].round(3)
    prior_weight = MIN_RATINGS_FOR_WEIGHTED_SCORE
    result["weighted_score"] = (
        (result["rating_count"] / (result["rating_count"] + prior_weight)) * result["avg_rating"].fillna(global_mean)
        + (prior_weight / (result["rating_count"] + prior_weight)) * global_mean
    ).round(3)
    result["genres_display"] = result["genre_list"].map(lambda genres: ", ".join(genres) if genres else "Genre unavailable")
    return result


def genre_rating_summary(movies: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-movie ratings by each listed genre."""
    expanded = movies.explode("genre_list").dropna(subset=["genre_list"])
    if expanded.empty:
        return pd.DataFrame(columns=["genre", "movie_count", "rating_count", "avg_rating"])
    return (
        expanded.groupby("genre_list", as_index=False)
        .agg(movie_count=("movieId", "nunique"), rating_count=("rating_count", "sum"), avg_rating=("avg_rating", "mean"))
        .rename(columns={"genre_list": "genre"})
        .sort_values("avg_rating", ascending=False)
    )


def filter_movies(
    movies: pd.DataFrame,
    search: str = "",
    genres: list[str] | None = None,
    years: tuple[int, int] | None = None,
    minimum_rating: float = 0.0,
    minimum_ratings: int = 0,
    sort_by: str = "Highest Rated",
) -> pd.DataFrame:
    """Apply the Discover page filters and supported result ordering."""
    result = movies.copy()
    if search.strip():
        result = result[result["title"].str.contains(search.strip(), case=False, regex=False, na=False)]
    if genres:
        selected_genres = set(genres)
        result = result[result["genre_list"].map(lambda values: bool(set(values) & selected_genres))]
    if years is not None:
        year_matches = result["year"].between(years[0], years[1])
        known_years = movies["year"].dropna()
        if not known_years.empty and years == (int(known_years.min()), int(known_years.max())):
            year_matches |= result["year"].isna()
        result = result[year_matches]
    if minimum_rating > 0:
        result = result[result["avg_rating"].ge(minimum_rating)]
    result = result[result["rating_count"].ge(minimum_ratings)]

    sort_options = {
        "Highest Rated": (("avg_rating", "rating_count"), (False, False)),
        "Most Rated": (("rating_count", "weighted_score"), (False, False)),
        "Newest": (("year", "rating_count"), (False, False)),
        "Oldest": (("year", "rating_count"), (True, False)),
        "Title A-Z": (("title", "rating_count"), (True, False)),
        "Title Z-A": (("title", "rating_count"), (False, False)),
    }
    if sort_by not in sort_options:
        raise ValueError(f"Unsupported movie sort: {sort_by}")
    columns, ascending = sort_options[sort_by]
    return result.sort_values(list(columns), ascending=list(ascending), na_position="last").reset_index(drop=True)


def recommend_similar_movies(movies: pd.DataFrame, movie_id: int, limit: int = 6) -> pd.DataFrame:
    """Rank films by genre Jaccard similarity, then CineSense Weighted Score."""
    chosen = movies[movies["movieId"] == movie_id]
    if chosen.empty:
        return movies.head(0)
    chosen_genres = set(chosen.iloc[0]["genre_list"])
    if not chosen_genres:
        return movies[movies["movieId"] != movie_id].sort_values("weighted_score", ascending=False).head(limit)

    candidates = movies[movies["movieId"] != movie_id].copy()
    def overlap(genres: list[str]) -> float:
        genre_set = set(genres)
        union = chosen_genres | genre_set
        return len(chosen_genres & genre_set) / len(union) if union else 0.0

    candidates["genre_similarity"] = candidates["genre_list"].map(overlap)
    return candidates[candidates["genre_similarity"] > 0].sort_values(
        ["genre_similarity", "weighted_score", "rating_count"], ascending=[False, False, False]
    ).head(limit)