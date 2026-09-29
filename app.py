"""CineSense: MovieLens ratings analysis with a dataset-grounded Gemini assistant."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from services.analytics_service import build_analytics
from services.data_service import (
    MIN_RATINGS_FOR_WEIGHTED_SCORE,
    DatasetError,
    add_movie_statistics,
    filter_movies,
    load_dataset,
    recommend_similar_movies,
)
from services.metadata_service import fetch_wikidata_metadata
from services.moviebot_service import (
    MOVIEBOT_UNAVAILABLE_MESSAGE,
    generate_answer,
    get_client,
    prepare_dataset_context,
)
from services.review_service import ReviewDatasetError, classify_review_text, load_linked_reviews


st.set_page_config(page_title="CineSense", page_icon="🎬", layout="wide")
DATA_DIR = Path(__file__).parent / "data"
PAGE_SIZE = 12
RATING_STARS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
DATA_CACHE_VERSION = "movielens-enrichment-review-v1"

st.markdown("""
<style>
:root {
  --cs-bg: #101214;
  --cs-panel: #191d20;
  --cs-panel-raised: #22282c;
  --cs-text: #f4f5f2;
  --cs-muted: #b1b9bc;
  --cs-gold: #f2c14e;
  --cs-teal: #62c6b7;
  --cs-border: #394247;
}
.stApp { background: var(--cs-bg); color: var(--cs-text); }
h1, h2, h3, h4, p, label, [data-testid="stMarkdownContainer"] { color: var(--cs-text); }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: var(--cs-muted); }
section[data-testid="stSidebar"] { background: var(--cs-panel); }
div[data-testid="stMetric"] { background: var(--cs-panel); border: 1px solid var(--cs-border); border-radius: 8px; padding: 14px; }
div[data-testid="stMetricLabel"] { color: var(--cs-muted); }
.cs-brand { font-size: 2rem; font-weight: 800; color: var(--cs-text); margin-bottom: 0; }
.cs-brand span { color: var(--cs-gold); }
.cs-subtitle { color: var(--cs-muted); margin-top: 0; }
.cs-rating { color: var(--cs-gold); font-weight: 700; }
button:focus-visible, input:focus-visible, textarea:focus-visible { outline: 2px solid var(--cs-teal) !important; outline-offset: 2px !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading and validating MovieLens data...")
def load_prepared_data(data_path: str, cache_version: str):
    movie_frame, rating_frame, tag_frame, link_frame = load_dataset(Path(data_path))
    result = add_movie_statistics(movie_frame, rating_frame)
    result = result.merge(link_frame[["movieId", "imdbId"]], on="movieId", how="left")
    result["imdb_url"] = result["imdbId"].map(
        lambda value: f"https://www.imdb.com/title/tt{str(value).zfill(7)}/" if pd.notna(value) and str(value) else ""
    )
    return result, rating_frame, tag_frame, build_analytics(result, rating_frame, tag_frame)


try:
    movies_df, ratings_df, tags_df, analytics = load_prepared_data(str(DATA_DIR), DATA_CACHE_VERSION)
except (DatasetError, FileNotFoundError, pd.errors.ParserError, OSError) as error:
    st.error("CineSense could not load its MovieLens dataset. Check that the required CSV files are present and valid.")
    st.caption(str(error))
    st.stop()

if "uploaded_movie_reviews" not in st.session_state:
    st.session_state["uploaded_movie_reviews"] = pd.DataFrame()
if "user_submitted_reviews" not in st.session_state:
    st.session_state["user_submitted_reviews"] = []
if "movie_metadata_by_id" not in st.session_state:
    st.session_state["movie_metadata_by_id"] = {}


def show_movie_table(frame: pd.DataFrame, key_prefix: str, limit: int = PAGE_SIZE) -> None:
    """Render a compact, source-faithful movie list with detail buttons."""
    visible = frame.head(limit)
    if visible.empty:
        st.info("No movies match these filters.")
        return
    for start in range(0, len(visible), 3):
        columns = st.columns(3)
        for column, (_, movie) in zip(columns, visible.iloc[start:start + 3].iterrows()):
            with column:
                with st.container(border=True):
                    st.markdown(f"#### {movie['title']}")
                    year_text = str(int(movie["year"])) if pd.notna(movie["year"]) else "Year not listed"
                    st.caption(f"{year_text} · {movie['genres_display']}")
                    if pd.notna(movie["avg_rating"]):
                        st.markdown(f"<span class='cs-rating'>{movie['avg_rating']:.2f}/5</span> · {int(movie['rating_count']):,} ratings", unsafe_allow_html=True)
                    else:
                        st.caption("No ratings in this dataset")
                    st.button(
                        "View Details",
                        key=f"{key_prefix}_{int(movie['movieId'])}",
                        width="stretch",
                        on_click=open_movie_detail,
                        args=(int(movie["movieId"]),),
                    )


def open_movie_detail(movie_id: int) -> None:
    st.session_state["detail_movie_id"] = movie_id


def open_review_analytics() -> None:
    st.session_state["active_page"] = "Review Analytics"


def show_linked_metadata(movie: pd.Series) -> None:
    movie_id = int(movie["movieId"])
    st.markdown("#### Additional metadata · Wikidata")
    st.caption("Fetched on demand via the MovieLens IMDb cross-reference. Each field is optional and may be missing.")
    cached_metadata = st.session_state["movie_metadata_by_id"].get(movie_id)
    if cached_metadata is None:
        if not st.button("Load Wikidata details", key=f"load_wikidata_{movie_id}"):
            st.info("Director, cast, runtime, description, and Commons poster are not supplied by MovieLens. Load linked open metadata to check for available fields.")
            return
        imdb_id = movie.get("imdbId")
        if pd.isna(imdb_id) or not str(imdb_id).strip():
            st.info("This MovieLens title has no IMDb cross-reference for a metadata lookup.")
            return
        release_year = int(movie["year"]) if pd.notna(movie["year"]) else None
        with st.spinner("Checking Wikidata for linked details..."):
            cached_metadata = fetch_wikidata_metadata(str(imdb_id), str(movie["title"]), release_year)
        if not cached_metadata.get("error"):
            st.session_state["movie_metadata_by_id"][movie_id] = cached_metadata

    if cached_metadata.get("error"):
        st.info(str(cached_metadata["error"]))
        return

    details_left, details_right = st.columns([1, 2])
    image_url = cached_metadata.get("image_url")
    with details_left:
        if image_url:
            st.image(image_url, width="stretch")
            if cached_metadata.get("image_page_url"):
                st.link_button("Image source and license", cached_metadata["image_page_url"])
        else:
            st.caption("No linked Wikimedia Commons poster image is available.")
    with details_right:
        directors = cached_metadata.get("directors", [])
        cast = cached_metadata.get("cast", [])
        runtime = cached_metadata.get("runtime_minutes")
        description = cached_metadata.get("description")
        st.write(f"**Director:** {', '.join(directors) if directors else 'Not available in Wikidata'}")
        st.write(f"**Cast:** {', '.join(cast) if cast else 'Not available in Wikidata'}")
        st.write(f"**Runtime:** {runtime:g} minutes" if runtime is not None else "**Runtime:** Not available in Wikidata")
        st.write(f"**Wikidata description:** {description}" if description else "**Wikidata description:** Not available")
        if cached_metadata.get("wikidata_url"):
            st.link_button("Metadata source · Wikidata", cached_metadata["wikidata_url"])
    plot_excerpt = cached_metadata.get("wikipedia_plot_excerpt")
    wikipedia_url = cached_metadata.get("wikipedia_url")
    if plot_excerpt and wikipedia_url:
        st.markdown("#### Plot excerpt · Wikipedia")
        st.write(plot_excerpt)
        st.caption("Wikipedia contributors · CC BY-SA 4.0. Excerpt reproduced from the linked article; verify details at the source.")
        st.link_button("Wikipedia plot source", wikipedia_url)
        st.link_button("Wikipedia CC BY-SA 4.0 license", "https://creativecommons.org/licenses/by-sa/4.0/")
    else:
        st.caption("No linked Wikipedia plot section is available for this movie.")


def show_movie_detail(movie: pd.Series) -> None:
    if st.button("Close Details", key=f"close_movie_details_{int(movie['movieId'])}"):
        st.session_state.pop("detail_movie_id", None)
        st.rerun()
    st.divider()
    st.markdown(f"## {movie['title']}")
    year_text = str(int(movie["year"])) if pd.notna(movie["year"]) else "Information not available in the dataset"
    st.write(f"**Release year:** {year_text}")
    st.write(f"**Genres:** {movie['genres_display']}")
    st.caption("Plot summaries are not included in MovieLens or the linked Wikidata fields currently used by CineSense.")
    show_linked_metadata(movie)

    metric_columns = st.columns(3)
    metric_columns[0].metric("MovieLens average rating", f"{movie['avg_rating']:.2f}/5" if pd.notna(movie["avg_rating"]) else "N/A")
    metric_columns[1].metric("MovieLens ratings", f"{int(movie['rating_count']):,}")
    metric_columns[2].metric("CineSense Weighted Score", f"{movie['weighted_score']:.3f}")
    if movie.get("imdb_url"):
        st.link_button("Open linked IMDb title", movie["imdb_url"])
        st.caption("The IMDb ID is supplied by MovieLens. Ratings shown here are from MovieLens, not IMDb.")

    movie_ratings = ratings_df[ratings_df["movieId"] == int(movie["movieId"])]
    distribution = movie_ratings["rating"].value_counts().reindex(RATING_STARS, fill_value=0).rename_axis("Stars").to_frame("Ratings")
    tags_for_movie = tags_df[tags_df["movieId"] == int(movie["movieId"])]
    left, right = st.columns(2)
    with left:
        st.markdown("#### Rating distribution")
        st.bar_chart(distribution.reset_index(), x="Stars", y="Ratings", x_label="Star rating", y_label="Number of ratings")
        st.caption("Each bar counts MovieLens users' submitted half-star ratings for this title.")
    with right:
        st.markdown("#### User-applied tags")
        if tags_for_movie.empty:
            st.info("No user tags are available for this movie in the dataset.")
        else:
            st.dataframe(
                tags_for_movie["tag"].value_counts().head(12).rename_axis("Tag").to_frame("Applications"),
                width="stretch",
            )
            st.caption("Tags are user-applied labels, not review text or verified plot facts.")

    movie_id = int(movie["movieId"])
    uploaded_reviews = st.session_state["uploaded_movie_reviews"]
    uploaded_for_movie = uploaded_reviews[uploaded_reviews["movieId"] == movie_id] if not uploaded_reviews.empty else pd.DataFrame()
    visitor_reviews = [review for review in st.session_state["user_submitted_reviews"] if review["movieId"] == movie_id]

    st.markdown("#### Written reviews")
    if not visitor_reviews and uploaded_for_movie.empty:
        st.markdown("#### MovieLens rating response")
        st.caption("The bundled source has no written reviews for this title. These are actual star-rating groups, not text sentiment.")
        total_ratings = len(movie_ratings)
        rating_groups = (
            ("High · 4–5 stars", int(movie_ratings["rating"].ge(4.0).sum())),
            ("Mid · 3–3.5 stars", int(movie_ratings["rating"].between(3.0, 3.5).sum())),
            ("Low · 0.5–2.5 stars", int(movie_ratings["rating"].le(2.5).sum())),
        )
        response_columns = st.columns(3)
        for column, (label, count) in zip(response_columns, rating_groups):
            share = count / total_ratings if total_ratings else 0.0
            column.metric(label, f"{count:,}", f"{share:.0%} of {total_ratings:,} ratings")
        st.button(
            "Add linked written reviews",
            key=f"add_reviews_{movie_id}",
            on_click=open_review_analytics,
        )
    else:
        if visitor_reviews:
            st.markdown("##### CineSense visitor-submitted")
            for review in reversed(visitor_reviews):
                with st.container(border=True):
                    st.text(f"{review['reviewer']} · {review['rating']}/5 stars · {review['submitted_at']}")
                    st.text(review["review_text"])
                    st.caption(f"VADER sentiment estimate: {review['sentiment']}. Visible for this session only.")

        if not uploaded_for_movie.empty:
            source_label = st.session_state.get("review_dataset_source", "User-provided review dataset")
            st.markdown("##### Uploaded review data")
            st.caption(f"Source: {source_label}. Labels are source-provided or VADER estimates.")
            counts = uploaded_for_movie["sentiment"].value_counts().reindex(["Positive", "Neutral", "Negative"], fill_value=0)
            st.bar_chart(counts.rename_axis("Sentiment").to_frame("Reviews"), x_label="Sentiment", y_label="Review count")
            st.dataframe(
                uploaded_for_movie[["sentiment", "review_text"]].head(8),
                width="stretch",
                hide_index=True,
            )

    visitor_review_frame = pd.DataFrame(visitor_reviews)
    review_evidence = pd.concat(
        [frame for frame in (uploaded_for_movie, visitor_review_frame) if not frame.empty],
        ignore_index=True,
        sort=False,
    ) if not uploaded_for_movie.empty or visitor_reviews else pd.DataFrame()
    if not review_evidence.empty:
        if st.button("Generate AI review summary", key=f"review_summary_{movie_id}"):
            client = get_client()
            if client is None:
                st.warning("MovieBot needs GEMINI_API_KEY to summarize uploaded reviews.")
            else:
                question = f"Summarize the available written reviews for {movie['title']}. Report positive and negative themes only if the excerpts support them."
                review_sources = []
                if not uploaded_for_movie.empty:
                    review_sources.append(st.session_state.get("review_dataset_source", "User-provided review dataset"))
                if visitor_reviews:
                    review_sources.append("CineSense visitor-submitted reviews (current session)")
                source_label = "; ".join(review_sources)
                context, _, _ = prepare_dataset_context(
                    question,
                    movies_df,
                    ratings_df,
                    tags_df,
                    review_evidence,
                    source_label,
                )
                with st.spinner("Summarizing only reviews linked to this movie..."):
                    summary = generate_answer(client, question, context, [])
                st.session_state.setdefault("review_summaries", {})[movie_id] = summary
        summary = st.session_state.get("review_summaries", {}).get(movie_id)
        if summary:
            st.markdown("**AI-generated review analysis**")
            st.write(summary)

    with st.expander("Write a review"):
        st.caption("Your review appears immediately and remains in this Streamlit session. It is not part of MovieLens or permanently stored.")
        with st.form(f"write_review_form_{movie_id}", clear_on_submit=True):
            reviewer_name = st.text_input("Display name", max_chars=80, key=f"reviewer_name_{movie_id}")
            visitor_rating = st.select_slider("Your rating", options=[1, 2, 3, 4, 5], value=4, key=f"visitor_rating_{movie_id}")
            review_text = st.text_area("Your review", placeholder="Share what you thought of this movie", max_chars=2000, key=f"visitor_review_text_{movie_id}")
            submitted = st.form_submit_button("Post review", type="primary")
            if submitted:
                clean_text = review_text.strip()
                if not clean_text:
                    st.warning("Write a few words before posting your review.")
                else:
                    st.session_state["user_submitted_reviews"].append({
                        "movieId": movie_id,
                        "reviewer": reviewer_name.strip() or "Anonymous",
                        "rating": int(visitor_rating),
                        "review_text": clean_text,
                        "sentiment": classify_review_text(clean_text),
                        "submitted_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                    })
                    st.rerun()

    st.markdown("#### Similar movies")
    similar = recommend_similar_movies(movies_df, int(movie["movieId"]), limit=6)
    if similar.empty:
        st.info("There are no genre-overlap recommendations for this movie.")
    else:
        st.caption("Ranked by genre overlap, then CineSense Weighted Score and rating count.")
        show_movie_table(similar, key_prefix=f"similar_{int(movie['movieId'])}", limit=6)


def render_selected_movie_detail() -> None:
    movie_id = st.session_state.get("detail_movie_id")
    if movie_id is None:
        return
    selected = movies_df[movies_df["movieId"] == movie_id]
    if not selected.empty:
        show_movie_detail(selected.iloc[0])


def render_home() -> None:
    st.markdown("<p class='cs-brand'>🎬 Cine<span>Sense</span></p>", unsafe_allow_html=True)
    st.markdown("<p class='cs-subtitle'>AI-Powered Movie Discovery &amp; Ratings Analysis</p>", unsafe_allow_html=True)
    genre_summary = analytics["genre_summary"]
    popular_genre = str(genre_summary["rating_count"].idxmax()) if not genre_summary.empty else "N/A"
    mean_rating = float(ratings_df["rating"].mean()) if not ratings_df.empty else 0.0
    metric_columns = st.columns(4)
    metric_columns[0].metric("Movies", f"{len(movies_df):,}")
    metric_columns[1].metric("Average user rating", f"{mean_rating:.2f}/5")
    metric_columns[2].metric("Ratings", f"{len(ratings_df):,}")
    metric_columns[3].metric("Most-rated genre", popular_genre)
    quick_search = st.text_input("Search movies, genres, or years", placeholder="Try: Toy Story, Comedy, or 2024")
    if quick_search.strip():
        query = quick_search.strip().casefold()
        matches = movies_df[
            movies_df["title"].str.casefold().str.contains(query, regex=False, na=False)
            | movies_df["genres_display"].str.casefold().str.contains(query, regex=False, na=False)
            | movies_df["year"].astype("string").str.contains(query, regex=False, na=False)
        ].sort_values(["weighted_score", "rating_count"], ascending=False)
        st.markdown(f"#### {min(6, len(matches))} of {len(matches):,} matching movies")
        show_movie_table(matches, "home_search", limit=6)

    render_selected_movie_detail()

    left, right = st.columns(2)
    with left:
        st.markdown("### 🔥 Most rated")
        st.caption("Titles with the most individual MovieLens rating records.")
        show_movie_table(movies_df.sort_values(["rating_count", "weighted_score"], ascending=False), "home_popular", limit=6)
    with right:
        st.markdown("### ⭐ Highest CineSense Weighted Score")
        st.caption(f"Bayesian ranking, with prior strength m={MIN_RATINGS_FOR_WEIGHTED_SCORE}; not an IMDb score.")
        eligible = movies_df[movies_df["rating_count"] >= MIN_RATINGS_FOR_WEIGHTED_SCORE]
        show_movie_table(eligible.sort_values(["weighted_score", "rating_count"], ascending=False), "home_top", limit=6)

    st.markdown("### 🆕 Recent MovieLens releases")
    st.caption("Newest release years available in this dated MovieLens snapshot; no current-release claim is implied.")
    recent = movies_df.dropna(subset=["year"]).sort_values(["year", "rating_count"], ascending=False)
    show_movie_table(recent, "home_recent", limit=6)

    st.markdown("### 🎭 Popular genres")
    popular = genre_summary.sort_values("rating_count", ascending=False).head(10)
    popular_plot = popular.reset_index()
    st.bar_chart(popular_plot, x="rating_count", y="Genre", horizontal=True, x_label="Number of ratings", y_label="Genre")
    if not popular.empty:
        st.caption(f"{popular.index[0]} has the most rating records among the displayed genres in this dataset.")


def render_discovery() -> None:
    st.title("🔎 Discover Movies")
    detail_id = st.session_state.get("detail_movie_id")
    if detail_id is not None:
        selected = movies_df[movies_df["movieId"] == detail_id]
        if not selected.empty:
            show_movie_detail(selected.iloc[0])
            return

    st.caption("Filter the MovieLens catalog and its observed rating records. Cast, director, runtime, and synopsis are not provided in these files.")
    filter_columns = st.columns([2, 1, 1, 1])
    search = filter_columns[0].text_input("Title search", placeholder="Search movie titles", key="discover_search")
    genres = sorted({genre for values in movies_df["genre_list"] for genre in values})
    genre_filter = filter_columns[1].multiselect("Genre", genres, key="discover_genres")
    years = movies_df["year"].dropna().astype(int)
    year_min, year_max = (int(years.min()), int(years.max())) if not years.empty else (1874, 2018)
    if year_min == year_max:
        year_filter = (year_min, year_max)
        filter_columns[2].caption(f"Only year: {year_min}")
    else:
        year_filter = filter_columns[2].slider("Release year", year_min, year_max, (year_min, year_max), key="discover_years")
    minimum_rating = filter_columns[3].slider("Minimum average rating", 0.0, 5.0, 0.0, 0.5, key="discover_rating")

    second_row = st.columns([1, 2, 2])
    minimum_ratings = second_row[0].number_input("Minimum rating count", min_value=0, value=0, step=10, key="discover_min_votes")
    sort_choice = second_row[1].selectbox(
        "Sort results",
        ["Highest Rated", "Most Rated", "Newest", "Oldest", "Title A-Z", "Title Z-A"],
        key="discover_sort",
    )
    second_row[2].caption("Rating count means individual MovieLens rating events; it is not a count of written reviews.")

    result = filter_movies(
        movies_df,
        search=search,
        genres=genre_filter,
        years=year_filter,
        minimum_rating=minimum_rating,
        minimum_ratings=int(minimum_ratings),
        sort_by=sort_choice,
    )
    st.caption(f"Full source catalog: {len(movies_df):,} movies")

    filter_signature = (search.strip().casefold(), tuple(sorted(genre_filter)), year_filter, minimum_rating, minimum_ratings, sort_choice)
    if st.session_state.get("discover_filter_signature") != filter_signature:
        st.session_state["discover_page"] = 1
        st.session_state["discover_filter_signature"] = filter_signature
    page_count = max(1, (len(result) + PAGE_SIZE - 1) // PAGE_SIZE)
    page_number = st.number_input("Page", min_value=1, max_value=page_count, step=1, key="discover_page")
    page_result = result.iloc[(int(page_number) - 1) * PAGE_SIZE:int(page_number) * PAGE_SIZE]
    st.markdown(f"### Showing {len(page_result):,} of {len(result):,} matching movies")
    show_movie_table(page_result, f"discover_page_{page_number}", limit=PAGE_SIZE)

    if not result.empty:
        detail_options = page_result["movieId"].tolist()
        selected_id = st.selectbox(
            "Open a movie detail view",
            detail_options,
            format_func=lambda movie_id: movies_df.loc[movies_df["movieId"] == movie_id, "title"].iloc[0],
            key="discover_detail_select",
        )
        if st.button("View selected movie details", key="discover_detail_button"):
            st.session_state["detail_movie_id"] = int(selected_id)
            st.rerun()


def render_analytics() -> None:
    st.title("📊 Review Analytics")
    st.caption("MovieLens supplies star ratings and user tags, not written reviews. Rating-derived sentiment below is a proxy; upload a properly linked review CSV for text-based analysis.")
    rating_dist = analytics["rating_distribution"]
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### Rating distribution")
        st.bar_chart(rating_dist.reset_index(), x="Star rating", y="Ratings", x_label="Stars", y_label="Rating records")
        st.caption("Counts of every observed half-star rating in MovieLens ratings.csv.")
    with col_b:
        st.markdown("### Average rating by genre")
        genre_stats = analytics["genre_summary"]
        genre_plot = genre_stats.reset_index()
        st.bar_chart(genre_plot, x="average_rating", y="Genre", horizontal=True, x_label="Average user rating (out of 5)", y_label="Genre")
        st.caption("Each rating is associated with every listed genre for that movie; multi-genre films therefore contribute to multiple groups.")

    st.markdown("### Rating-derived sentiment proxy")
    rating_sentiment = analytics["rating_sentiment_proxy"]
    st.bar_chart(rating_sentiment, x_label="Label", y_label="MovieLens ratings")
    st.caption("Not review-text sentiment: ratings ≥4.0 are Positive, ratings 3.0–3.5 are Neutral, and ratings ≤2.5 are Negative. These are analysis thresholds applied to MovieLens stars.")

    st.markdown("### Rating activity over time")
    yearly = analytics["ratings_by_year"]
    trend_left, trend_right = st.columns(2)
    with trend_left:
        st.line_chart(yearly.reset_index(), x="year", y="rating_count", x_label="Year", y_label="Number of ratings")
    with trend_right:
        st.line_chart(yearly.reset_index(), x="year", y="average_rating", x_label="Year", y_label="Average rating")
    st.caption("Year comes from each rating timestamp, not the movie's release year. The snapshot contains activity through September 2018.")

    st.markdown("### Most-applied user tags")
    tag_counts = analytics["common_tags"]
    tag_plot = tag_counts.reset_index()
    st.bar_chart(tag_plot, x="Applications", y="User tag", horizontal=True, x_label="Tag applications", y_label="Tag")
    st.caption("Tags are short, user-applied labels and are not reviews or opinions.")

    st.markdown("### Add a movie-linked written review dataset")
    st.write("Upload only review text you are permitted to use. The file is processed in this session and is not bundled or saved to disk.")
    st.code("Required: review_text and movieId (or imdbId). Optional: sentiment (Positive/Neutral/Negative).", language="text")
    review_source = st.text_input(
        "Review dataset source or citation",
        placeholder="Name the source and its reuse terms",
        key="review_dataset_source",
    )
    review_upload = st.file_uploader("Upload linked review CSV", type=["csv"], key="linked_review_csv")
    if review_upload is not None:
        upload_bytes = review_upload.getvalue()
        upload_key = (review_upload.name, hash(upload_bytes))
        if st.session_state.get("review_upload_key") != upload_key:
            try:
                loaded_reviews = load_linked_reviews(upload_bytes, movies_df[["movieId", "imdbId"]])
                st.session_state["uploaded_movie_reviews"] = loaded_reviews
                st.session_state["review_upload_key"] = upload_key
                st.success(f"Loaded {len(loaded_reviews):,} movie-linked review rows from this upload.")
            except ReviewDatasetError as error:
                st.error(str(error))

    uploaded_reviews = st.session_state["uploaded_movie_reviews"]
    if uploaded_reviews.empty:
        st.info("No written review text is bundled. The MovieLens rating sentiment proxy above works without an upload.")
    else:
        sentiment_counts = uploaded_reviews["sentiment"].value_counts().reindex(["Positive", "Neutral", "Negative"], fill_value=0).rename_axis("Sentiment").to_frame("Reviews")
        review_metrics = st.columns(3)
        review_metrics[0].metric("Linked reviews", f"{len(uploaded_reviews):,}")
        review_metrics[1].metric("Movies with reviews", f"{uploaded_reviews['movieId'].nunique():,}")
        review_metrics[2].metric("Review source", review_source.strip() or "Not specified")
        st.bar_chart(sentiment_counts, x_label="Sentiment", y_label="Uploaded reviews")
        review_preview = uploaded_reviews[["movieId", "sentiment", "review_text"]].head(10).copy()
        review_preview.insert(1, "Movie", review_preview["movieId"].map(movies_df.set_index("movieId")["title"]))
        st.dataframe(review_preview.drop(columns="movieId"), width="stretch", hide_index=True)
        st.caption("Uploaded labels are retained when provided; unlabeled text receives a VADER polarity estimate. Select a movie in Discover to see its matching reviews and optional AI summary.")
        if st.button("Clear uploaded review data", key="clear_review_upload"):
            st.session_state["uploaded_movie_reviews"] = pd.DataFrame()
            st.session_state.pop("review_upload_key", None)
            st.rerun()

    st.markdown("### Genre data table")
    st.dataframe(genre_stats, width="stretch")
    if not genre_stats.empty:
        best = genre_stats["average_rating"].idxmax()
        most_rated = genre_stats["rating_count"].idxmax()
        st.info(f"In this dataset, {best} has the highest average rating ({genre_stats.loc[best, 'average_rating']:.2f}/5), while {most_rated} has the most rating records ({int(genre_stats.loc[most_rated, 'rating_count']):,}).")


def render_moviebot() -> None:
    st.title("🤖 MovieBot")
    st.write("Ask questions about MovieLens titles, genres, release years, rating patterns, user tags, and similar-movie recommendations.")
    if st.session_state["uploaded_movie_reviews"].empty:
        st.info("MovieLens does not include written reviews. Upload a legally usable, movie-ID-linked review CSV in Review Analytics to enable title-specific review analysis. Other missing fields are never invented.")
    else:
        review_source = st.session_state.get("review_dataset_source", "User-provided review dataset") or "User-provided review dataset (source not specified)"
        st.info(f"MovieBot can use the {len(st.session_state['uploaded_movie_reviews']):,} uploaded, movie-linked reviews. Source label: {review_source}. Other missing fields are never invented.")
    with st.expander("Example prompts"):
        st.markdown("""
- Which movies have the highest ratings?
- What are the most popular horror movies?
- Show me movies from 2024 with ratings above 4.
- What genres have the highest average ratings?
- Which movies have the most ratings?
- Recommend movies similar to Toy Story.
- What do MovieLens users tag this movie with?
- Summarize the reviews for The Matrix. (The dataset has no review text.)
- Help me understand this dataset.
- Which movies have director Christopher Nolan? (Director is not a source field.)
""")

    client = get_client()
    if client is None:
        st.warning("MovieBot is temporarily unavailable. Add GEMINI_API_KEY in Streamlit secrets to enable Gemini; browsing and analytics remain available.")

    if "moviebot_messages" not in st.session_state:
        st.session_state["moviebot_messages"] = []
    for message in st.session_state["moviebot_messages"][-10:]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    question = st.chat_input("Ask a question about this dataset...", disabled=client is None)
    if question:
        history = st.session_state["moviebot_messages"][-6:]
        st.session_state["moviebot_messages"].append({"role": "user", "content": question})
        context, relevant_movies, filters = prepare_dataset_context(question, movies_df, ratings_df, tags_df)
        answer_cache = st.session_state.setdefault("moviebot_answer_cache", {})
        cache_key = (question, context, tuple((item["role"], item["content"]) for item in history))
        if cache_key in answer_cache:
            answer = answer_cache[cache_key]
        else:
            with st.spinner("Filtering MovieLens records and preparing a grounded answer..."):
                answer = generate_answer(client, question, context, history)
            if answer != MOVIEBOT_UNAVAILABLE_MESSAGE:
                answer_cache[cache_key] = answer
                if len(answer_cache) > 30:
                    answer_cache.pop(next(iter(answer_cache)))
        st.session_state["moviebot_messages"].append({"role": "assistant", "content": answer})
        st.rerun()

    if st.session_state["moviebot_messages"]:
        if st.button("Clear chat", key="clear_moviebot"):
            st.session_state["moviebot_messages"] = []
            st.rerun()


with st.sidebar:
    st.markdown("<p class='cs-brand'>🎬 Cine<span>Sense</span></p>", unsafe_allow_html=True)
    st.caption("MovieLens ratings · Pandas analytics · Gemini MovieBot")
    st.divider()
    st.metric("Movies loaded", f"{len(movies_df):,}")
    st.metric("Ratings analyzed", f"{len(ratings_df):,}")
    available_years = movies_df["year"].dropna()
    release_range = f"{int(available_years.min())}–{int(available_years.max())}" if not available_years.empty else "unavailable"
    st.caption(f"Release years: {release_range}")

active_page = st.radio(
    "Navigate CineSense",
    ["Home", "Discover Movies", "Review Analytics", "MovieBot"],
    key="active_page",
    horizontal=True,
    label_visibility="collapsed",
)
st.divider()

if active_page == "Home":
    render_home()
elif active_page == "Discover Movies":
    render_discovery()
elif active_page == "Review Analytics":
    render_analytics()
else:
    render_moviebot()
