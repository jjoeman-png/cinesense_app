import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class AppNavigationTests(unittest.TestCase):
    def test_movie_details_stay_on_the_originating_page(self):
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path), default_timeout=45).run()
        app.button(key="home_popular_356").click().run()
        self.assertEqual(app.session_state["active_page"], "Home")
        self.assertEqual(app.session_state["detail_movie_id"], 356)
        self.assertFalse(app.exception)
        response_labels = {metric.label for metric in app.metric}
        self.assertTrue({"High · 4–5 stars", "Mid · 3–3.5 stars", "Low · 0.5–2.5 stars"}.issubset(response_labels))

        app.button(key="close_movie_details_356").click().run()
        self.assertIsNone(app.session_state.get("detail_movie_id"))

        app.button(key="home_popular_356").click().run()
        app.button(key="add_reviews_356").click().run()
        self.assertEqual(app.session_state["active_page"], "Review Analytics")
        app.radio(key="active_page").set_value("Home").run()
        app.button(key="close_movie_details_356").click().run()

        app.radio(key="active_page").set_value("Discover Movies").run()
        details_button = next(
            button for button in app.button
            if button.key and button.key.startswith("discover_page_")
        )
        app.button(key=details_button.key).click().run()
        self.assertEqual(app.session_state["active_page"], "Discover Movies")
        self.assertEqual(app.session_state["detail_movie_id"], 53)
        self.assertFalse(app.exception)
        detail_metrics = {metric.label for metric in app.metric}
        self.assertIn("MovieLens average rating", detail_metrics)
        self.assertIn("CineSense Weighted Score", detail_metrics)
        self.assertTrue(any(button.label == "Close Details" for button in app.button))
        self.assertFalse(any("Showing" in item.value for item in app.markdown))
        self.assertFalse(any(button.key and button.key.startswith("discover_page_") for button in app.button))
        app.button(key=f"close_movie_details_{app.session_state['detail_movie_id']}").click().run()
        self.assertTrue(any(item.key == "discover_search" for item in app.text_input))
        self.assertFalse(app.exception)
        self.assertEqual(
            app.radio(key="active_page").options,
            ["Home", "Discover Movies", "Review Analytics", "MovieBot"],
        )

        app.radio(key="active_page").set_value("Home").run()
        app.button(key="home_popular_356").click().run()
        app.text_input(key="reviewer_name_356").set_value("Alex").run()
        app.select_slider(key="visitor_rating_356").set_value(5).run()
        app.text_area(key="visitor_review_text_356").set_value("A wonderful, heartfelt movie.").run()
        app.button(key="FormSubmitter:write_review_form_356-Post review").click().run()
        self.assertEqual(len(app.session_state["user_submitted_reviews"]), 1)
        self.assertEqual(app.session_state["user_submitted_reviews"][0]["movieId"], 356)
        self.assertEqual(app.session_state["user_submitted_reviews"][0]["sentiment"], "Positive")
        displayed_text = [element.value for element in app.get("text")]
        self.assertIn("A wonderful, heartfelt movie.", displayed_text)
        self.assertFalse(app.exception)


if __name__ == "__main__":
    unittest.main()