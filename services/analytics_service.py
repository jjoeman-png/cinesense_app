"""Pandas summaries for the Review Analytics view."""

from __future__ import annotations

import pandas as pd


def build_analytics(movies: pd.DataFrame, ratings: pd.DataFrame, tags: pd.DataFrame) -> dict[str, pd.DataFrame]:
    rating_distribution = ratings["rating"].value_counts().sort_index().rename_axis("Star rating").to_frame("Ratings")
    rating_sentiment = ratings["rating"].map(
        lambda value: "Positive" if value >= 4.0 else "Negative" if value <= 2.5 else "Neutral"
    ).value_counts().reindex(["Positive", "Neutral", "Negative"], fill_value=0).rename_axis("Rating-derived label").to_frame("Ratings")

    ratings_by_year = (
        ratings.assign(year=ratings["rated_at"].dt.year)
        .groupby("year", as_index=True)
        .agg(rating_count=("rating", "size"), average_rating=("rating", "mean"))
        .round(3)
    )

    movie_genres = movies[["movieId", "genre_list"]].explode("genre_list").dropna(subset=["genre_list"])
    genre_movie_counts = movie_genres.groupby("genre_list")["movieId"].nunique()
    genre_ratings = ratings.merge(movie_genres, on="movieId", how="inner")
    genre_summary = (
        genre_ratings.groupby("genre_list")
        .agg(rating_count=("rating", "size"), average_rating=("rating", "mean"))
        .round(3)
    )
    genre_summary["movie_count"] = genre_movie_counts
    genre_summary = genre_summary.rename_axis("Genre").sort_values("average_rating", ascending=False)

    ratings_over_time = (
        ratings.set_index("rated_at")["rating"]
        .resample("YS")
        .agg(["count", "mean"])
        .rename(columns={"count": "Ratings", "mean": "Average rating"})
        .round(3)
    )

    common_tags = tags["tag"].value_counts().head(20).rename_axis("User tag").to_frame("Applications")
    return {
        "rating_distribution": rating_distribution,
        "rating_sentiment_proxy": rating_sentiment,
        "ratings_by_year": ratings_by_year,
        "genre_summary": genre_summary,
        "ratings_over_time": ratings_over_time,
        "common_tags": common_tags,
    }