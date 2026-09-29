"""Load a user-provided, movie-linked review CSV without bundling review text."""

from __future__ import annotations

import io
import re

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


MAX_REVIEW_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_REVIEW_ROWS = 20_000
_SENTIMENT = SentimentIntensityAnalyzer()


class ReviewDatasetError(ValueError):
    """Raised when an uploaded movie review dataset is not safely joinable."""


def _normalize_imdb_id(value: object) -> str:
    digits = re.sub(r"\D", "", str(value))
    return digits.zfill(7) if digits else ""


def classify_review_text(text: str) -> str:
    compound = _SENTIMENT.polarity_scores(text)["compound"]
    if compound >= 0.05:
        return "Positive"
    if compound <= -0.05:
        return "Negative"
    return "Neutral"


def load_linked_reviews(file_bytes: bytes, links: pd.DataFrame) -> pd.DataFrame:
    """Validate ID-linked review text and use source labels or VADER polarity."""
    if len(file_bytes) > MAX_REVIEW_UPLOAD_BYTES:
        raise ReviewDatasetError("Review CSV exceeds the 10 MB upload limit.")
    try:
        reviews = pd.read_csv(io.BytesIO(file_bytes), dtype={"imdbId": "string", "review_text": "string"})
    except Exception as error:
        raise ReviewDatasetError("The uploaded file could not be read as a CSV.") from error
    if len(reviews) > MAX_REVIEW_ROWS:
        raise ReviewDatasetError(f"Review CSV exceeds the {MAX_REVIEW_ROWS:,}-row processing limit.")
    if "review_text" not in reviews.columns:
        raise ReviewDatasetError("The CSV must contain a review_text column.")
    if not ({"movieId", "imdbId"} & set(reviews.columns)):
        raise ReviewDatasetError("The CSV must contain movieId or imdbId so reviews can be linked to movies.")

    if "movieId" in reviews.columns:
        reviews["movieId"] = pd.to_numeric(reviews["movieId"], errors="coerce")
        reviews = reviews.dropna(subset=["movieId"])
        reviews = reviews[reviews["movieId"].mod(1).eq(0)].copy()
        reviews["movieId"] = reviews["movieId"].astype("int64")
    else:
        imdb_map = links.dropna(subset=["imdbId"]).copy()
        imdb_map["imdb_join_key"] = imdb_map["imdbId"].map(_normalize_imdb_id)
        reviews["imdb_join_key"] = reviews["imdbId"].map(_normalize_imdb_id)
        reviews = reviews.merge(imdb_map[["movieId", "imdb_join_key"]], on="imdb_join_key", how="inner")

    valid_ids = set(links["movieId"].astype("int64"))
    reviews = reviews[reviews["movieId"].isin(valid_ids)].copy()
    reviews["review_text"] = reviews["review_text"].fillna("").str.strip()
    reviews = reviews[reviews["review_text"].str.len().between(1, 10_000)]
    reviews = reviews.drop_duplicates(["movieId", "review_text"], keep="first")
    if reviews.empty:
        raise ReviewDatasetError("No non-empty reviews could be matched to MovieLens movie IDs.")

    if "sentiment" in reviews.columns:
        labels = reviews["sentiment"].fillna("").str.strip().str.casefold()
        normalized = labels.map({
            "positive": "Positive", "pos": "Positive", "1": "Positive",
            "negative": "Negative", "neg": "Negative", "0": "Negative",
            "neutral": "Neutral",
        })
        missing_labels = normalized.isna()
        normalized.loc[missing_labels] = reviews.loc[missing_labels, "review_text"].map(classify_review_text)
        reviews["sentiment"] = normalized
        reviews["sentiment_method"] = "Uploaded labels; VADER for missing labels"
    else:
        reviews["sentiment"] = reviews["review_text"].map(classify_review_text)
        reviews["sentiment_method"] = "VADER automated text analysis"

    return reviews.reset_index(drop=True)