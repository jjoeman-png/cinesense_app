import json
from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from services.analytics_service import build_analytics
from services.data_service import (
    add_movie_statistics,
    DatasetError,
    filter_movies,
    load_dataset,
    recommend_similar_movies,
)
from services.moviebot_service import prepare_dataset_context
from services.metadata_service import (
    _fetch_wikipedia_plot,
    _metadata_from_wikipedia,
    _normalized_article_title,
    fetch_wikidata_metadata,
)
from services.review_service import ReviewDatasetError, load_linked_reviews


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


class MovieLensServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source_movies, cls.ratings, cls.tags, cls.links = load_dataset(DATA_DIR)
        cls.movies = add_movie_statistics(cls.source_movies, cls.ratings)

    def test_bundled_snapshot_counts_and_types(self):
        self.assertEqual(len(self.movies), 9742)
        self.assertEqual(len(self.ratings), 100836)
        self.assertEqual(len(self.tags), 3683)
        self.assertEqual(len(self.links), 9742)
        self.assertTrue(self.ratings["rating"].between(0.5, 5.0).all())
        self.assertTrue(self.movies["genre_list"].map(lambda values: isinstance(values, list)).all())

    def test_data_cleaning_and_aggregates(self):
        toy_story = self.movies[self.movies["title"] == "Toy Story"].iloc[0]
        self.assertEqual(int(toy_story["year"]), 1995)
        self.assertIn("Animation", toy_story["genre_list"])
        self.assertEqual(int(toy_story["rating_count"]), 215)
        self.assertAlmostEqual(float(toy_story["avg_rating"]), 3.921, places=3)

    def test_analytics_cover_source_rating_rows(self):
        analytics = build_analytics(self.movies, self.ratings, self.tags)
        self.assertEqual(int(analytics["rating_distribution"]["Ratings"].sum()), len(self.ratings))
        self.assertEqual(int(analytics["ratings_by_year"]["rating_count"].sum()), len(self.ratings))
        self.assertEqual(len(analytics["genre_summary"]), 19)
        self.assertFalse(analytics["common_tags"].empty)

    def test_filter_search_genre_year_and_rating(self):
        bracket_titles = filter_movies(self.movies, search="[")
        self.assertTrue(bracket_titles["title"].str.contains("[", regex=False).all())
        self.assertTrue(filter_movies(self.movies, search="CineSenseNoSuchMovieXYZ").empty)

        horror = filter_movies(self.movies, genres=["Horror"])
        self.assertGreater(len(horror), 0)
        self.assertTrue(horror["genre_list"].map(lambda values: "Horror" in values).all())

        releases_1995 = filter_movies(self.movies, years=(1995, 1995))
        self.assertTrue(releases_1995["year"].eq(1995).all())
        highly_rated = filter_movies(self.movies, minimum_rating=4.0)
        self.assertTrue(highly_rated["avg_rating"].ge(4.0).all())
        popular = filter_movies(self.movies, minimum_ratings=100)
        self.assertTrue(popular["rating_count"].ge(100).all())

    def test_all_sort_modes(self):
        for sort in ("Highest Rated", "Most Rated", "Newest", "Oldest", "Title A-Z", "Title Z-A"):
            with self.subTest(sort=sort):
                self.assertEqual(len(filter_movies(self.movies, sort_by=sort)), len(self.movies))
        az = filter_movies(self.movies, sort_by="Title A-Z")["title"].tolist()
        za = filter_movies(self.movies, sort_by="Title Z-A")["title"].tolist()
        self.assertEqual(az, sorted(self.movies["title"].tolist()))
        self.assertEqual(za, sorted(self.movies["title"].tolist(), reverse=True))
        self.assertEqual(
            filter_movies(self.movies, sort_by="Highest Rated").iloc[0]["avg_rating"],
            self.movies["avg_rating"].max(),
        )

    def test_invalid_sort_is_rejected(self):
        with self.assertRaises(ValueError):
            filter_movies(self.movies, sort_by="Random")

    def test_newest_and_oldest_sort_directions(self):
        newest_years = filter_movies(self.movies, sort_by="Newest")["year"].dropna().astype(int).tolist()
        oldest_years = filter_movies(self.movies, sort_by="Oldest")["year"].dropna().astype(int).tolist()
        self.assertEqual(newest_years, sorted(newest_years, reverse=True))
        self.assertEqual(oldest_years, sorted(oldest_years))

    def test_missing_dataset_file_fails_with_friendly_error_type(self):
        with tempfile.TemporaryDirectory() as missing_directory:
            with self.assertRaisesRegex(DatasetError, "movies.csv"):
                load_dataset(Path(missing_directory))

    def test_malformed_dataset_schema_fails_clearly(self):
        with tempfile.TemporaryDirectory() as malformed_directory:
            pd.DataFrame({"movieId": [1]}).to_csv(Path(malformed_directory) / "movies.csv", index=False)
            with self.assertRaisesRegex(DatasetError, "missing columns"):
                load_dataset(Path(malformed_directory))

    def test_similar_movie_recommendations(self):
        toy_story = self.movies[self.movies["title"] == "Toy Story"].iloc[0]
        suggestions = recommend_similar_movies(self.movies, int(toy_story["movieId"]))
        self.assertFalse(suggestions.empty)
        self.assertNotIn(int(toy_story["movieId"]), suggestions["movieId"].tolist())
        self.assertTrue(suggestions["genre_list"].map(lambda values: bool(set(values) & set(toy_story["genre_list"]))).all())

    def test_moviebot_example_prompt_context_is_bounded_and_grounded(self):
        prompts = [
            "Which movies have the highest ratings?",
            "What are the most popular horror movies?",
            "Show me movies from 2024 with ratings above 4.",
            "What genres have the highest average ratings?",
            "Which movies have the most ratings?",
            "Recommend movies similar to Toy Story.",
            "What do MovieLens users tag Toy Story with?",
            "Summarize the reviews for The Matrix.",
            "Help me understand this dataset.",
            "Which movies have director Christopher Nolan?",
        ]
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                context, matches, _ = prepare_dataset_context(prompt, self.movies, self.ratings, self.tags)
                self.assertLessEqual(len(matches), 12)
                self.assertIn("MovieLens latest-small", context)
                self.assertIn("review text", context)

        _, horror_matches, horror_filter = prepare_dataset_context(prompts[1], self.movies, self.ratings, self.tags)
        self.assertEqual(horror_filter["genres"], ["Horror"])
        self.assertTrue(horror_matches["genre_list"].map(lambda values: "Horror" in values).all())

        _, year_rating_matches, applied = prepare_dataset_context(prompts[2], self.movies, self.ratings, self.tags)
        self.assertEqual(applied["year"], 2024)
        self.assertEqual(applied["minimum_average_rating"], 4.0)
        self.assertTrue(year_rating_matches["year"].eq(2024).all())
        self.assertTrue(year_rating_matches["avg_rating"].ge(4.0).all())

        _, similar, similar_filter = prepare_dataset_context(prompts[5], self.movies, self.ratings, self.tags)
        self.assertEqual(similar_filter["similar_to_title"], "Toy Story")
        self.assertFalse(similar["title"].eq("Toy Story").any())

        _, matrix, matrix_filter = prepare_dataset_context(prompts[7], self.movies, self.ratings, self.tags)
        self.assertEqual(matrix_filter["title"], "Matrix, The")
        self.assertEqual(matrix.iloc[0]["title"], "Matrix, The")

    def test_source_contains_no_review_text(self):
        self.assertFalse((DATA_DIR / "reviews.csv").exists())
        context, _, _ = prepare_dataset_context("Summarize the reviews for The Matrix.", self.movies, self.ratings, self.tags)
        self.assertIn("review text", context)
        self.assertIn("unavailable_fields", context)

    def test_uploaded_movie_linked_reviews_receive_vader_labels(self):
        review_csv = (
            b"movieId,review_text\n"
            b"1,This is a wonderful and joyful movie!\n"
            b"1,This is a boring and awful movie.\n"
        )
        reviews = load_linked_reviews(review_csv, self.links)
        self.assertEqual(reviews["sentiment"].tolist(), ["Positive", "Negative"])
        self.assertEqual(set(reviews["sentiment_method"]), {"VADER automated text analysis"})

    def test_uploaded_imdb_ids_join_and_keep_source_labels(self):
        review_csv = b"imdbId,review_text,sentiment\n0114709,A source-labeled review,positive\n"
        reviews = load_linked_reviews(review_csv, self.links)
        self.assertEqual(reviews.iloc[0]["movieId"], 1)
        self.assertEqual(reviews.iloc[0]["sentiment"], "Positive")
        self.assertIn("Uploaded labels", reviews.iloc[0]["sentiment_method"])

    def test_uploaded_reviews_without_movie_match_are_rejected(self):
        review_csv = b"movieId,review_text\n999999999,Unmatched review text\n"
        with self.assertRaisesRegex(ReviewDatasetError, "matched"):
            load_linked_reviews(review_csv, self.links)

    @patch("services.metadata_service.urllib.request.urlopen")
    def test_wikipedia_plot_excerpt_has_article_provenance(self, mock_urlopen):
        sections = {"parse": {"sections": [{"index": "3", "line": "Plot"}]}}
        plot_html = {"parse": {"text": {"*": "<div><p>A character follows the map.</p><sup>1</sup><p>The story ends.</p></div>"}}}
        mock_urlopen.side_effect = [
            BytesIO(json.dumps(sections).encode("utf-8")),
            BytesIO(json.dumps(plot_html).encode("utf-8")),
        ]
        result = _fetch_wikipedia_plot("https://en.wikipedia.org/wiki/Example_film")
        self.assertIn("A character follows the map.", result["wikipedia_plot_excerpt"])
        self.assertIn("The story ends.", result["wikipedia_plot_excerpt"])
        self.assertNotIn("1", result["wikipedia_plot_excerpt"])
        self.assertEqual(result["wikipedia_url"], "https://en.wikipedia.org/wiki/Example_film")

    def test_metadata_fallback_validates_imdb_id_and_resolves_labels(self):
        entity = {
            "claims": {
                "P345": [{"mainsnak": {"snaktype": "value", "datavalue": {"value": "tt0114709"}}}],
                "P57": [{"mainsnak": {"snaktype": "value", "datavalue": {"value": {"id": "Q100"}}}}],
                "P161": [{"mainsnak": {"snaktype": "value", "datavalue": {"value": {"id": "Q101"}}}}],
                "P2047": [{"mainsnak": {"snaktype": "value", "datavalue": {"value": {"amount": "+81", "unit": "http://www.wikidata.org/entity/Q7727"}}}}],
            }
        }
        responses = [
            {"query": {"search": [{"title": "Toy Story"}]}},
            {"query": {"pages": {"1": {"pageprops": {"wikibase_item": "Q171048"}}}}},
            {"entities": {"Q171048": entity}},
            {"entities": {
                "Q100": {"labels": {"en": {"value": "John Lasseter"}}},
                "Q101": {"labels": {"en": {"value": "Tom Hanks"}}},
            }},
        ]
        with patch("services.metadata_service._api_json", side_effect=responses), patch(
            "services.metadata_service._fetch_wikipedia_plot",
            return_value={"wikipedia_plot_excerpt": "Verified plot excerpt", "wikipedia_url": "https://en.wikipedia.org/wiki/Toy_Story"},
        ):
            result = _metadata_from_wikipedia("tt0114709", "Toy Story", 1995)
        self.assertEqual(result["directors"], ["John Lasseter"])
        self.assertEqual(result["cast"], ["Tom Hanks"])
        self.assertEqual(result["runtime_minutes"], 81)
        self.assertIn("Verified plot excerpt", result["wikipedia_plot_excerpt"])

        mismatched = [*responses[:2], {"entities": {"Q171048": {
            "claims": {"P345": [{"mainsnak": {"snaktype": "value", "datavalue": {"value": "tt9999999"}}}]}
        }}}]
        with patch("services.metadata_service._api_json", side_effect=mismatched):
            self.assertEqual(_metadata_from_wikipedia("tt0114709", "Toy Story", 1995), {})

    def test_metadata_fallback_matches_inverted_movie_titles(self):
        self.assertEqual(
            _normalized_article_title("Matrix, The"),
            _normalized_article_title("The Matrix (1999)"),
        )

    def test_sparql_failure_uses_wikipedia_metadata_fallback(self):
        fallback = {
            "wikidata_url": "https://www.wikidata.org/wiki/Q171048",
            "directors": ["John Lasseter"],
        }
        with patch("services.metadata_service.urllib.request.urlopen", side_effect=OSError("429 throttled")), patch(
            "services.metadata_service._metadata_from_wikipedia", return_value=fallback
        ) as fallback_lookup:
            result = fetch_wikidata_metadata("0114709", "Toy Story", 1995)
        fallback_lookup.assert_called_once_with("tt0114709", "Toy Story", 1995)
        self.assertEqual(result, fallback)

    def test_moviebot_includes_only_matching_uploaded_review_excerpts(self):
        reviews = load_linked_reviews(
            b"movieId,review_text\n1,Toy Story has delightful animation.\n2,Midnight Freeway is thrilling.\n",
            self.links,
        )
        context, _, _ = prepare_dataset_context(
            "Summarize the reviews for Toy Story.",
            self.movies,
            self.ratings,
            self.tags,
            reviews,
            "Test user-uploaded source",
        )
        self.assertIn("Toy Story has delightful animation", context)
        self.assertNotIn("Midnight Freeway is thrilling", context)
        self.assertIn("Test user-uploaded source", context)


if __name__ == "__main__":
    unittest.main()