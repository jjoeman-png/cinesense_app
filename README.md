# 🎬 CineSense

A movie discovery and review site — built with Streamlit and powered by Google
Gemini (GenAI). Browse and search a catalog of 2020–2026 releases, read and
post reviews, and chat with **MovieBot**, a floating assistant that can
search titles, recommend movies, answer questions about cast/rating/plot,
help you draft a review, and summarize what other viewers thought.

End users never see or enter an API key — MovieBot is powered by a key the
site owner configures once (see below).

## Project structure

```
cinesense_app/
├── app.py                        # Streamlit app
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   ├── config.toml               # dark theme
│   └── secrets.toml.example      # template — copy to secrets.toml locally
└── data/
    ├── movies.csv                # 30-title catalog, 2020–2026
    └── reviews.csv               # seed reviews
```

## 1. Get a free Gemini API key (one-time, for you as the site owner)

1. Go to <https://aistudio.google.com/apikey>.
2. Sign in with a Google account and click **Create API key** — free, no
   credit card required.
3. Copy the key. You will set it once as a secret (below); the app's
   visitors never see or type a key themselves.

## 2. Run locally

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# then edit .streamlit/secrets.toml and paste in your key
streamlit run app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`).
Browsing, filtering, and posting reviews all work even without a key;
only the MovieBot chat requires one.

## 3. Deploy to Streamlit Community Cloud (free)

1. Push this folder to a GitHub repository. **Do not commit
   `.streamlit/secrets.toml`** — it's already in `.gitignore`.
2. Go to <https://streamlit.io/cloud> and sign in with GitHub.
3. Click **New app**, pick the repo/branch, and set the main file to
   `app.py`.
4. Before (or right after) deploying, open **App settings → Secrets** and
   paste:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
5. Click **Deploy** (or **Save** then **Reboot app** if it's already
   running). Your app gets a public `https://<name>.streamlit.app` link —
   visitors use MovieBot immediately, with no setup on their end.

## Features

- **Browse & search** — sidebar filters by genre, year (2020–2026), and
  free-text search across title / cast / director; sortable by rating,
  newest, or title.
- **Home** — Trending Now and Top Rated rows.
- **Movie detail pages** — synopsis, cast, director, runtime, star rating,
  and every review.
- **Write a review** — star rating + text, posted instantly to the page.
- **💬 MovieBot** (floating button, bottom-right) — five core functions:
  1. **Movie search** — "Show me movies starring Devon Cross"
  2. **Recommendations** — "Recommend me a horror movie"
  3. **Movie information** — "What's the rating of Iron Meridian?"
  4. **Review-writing assistance** — "Help me review a movie I just watched"
  5. **Review summarization** — "What do people generally think of Cabin Fourteen?"

  MovieBot only discusses movies in the catalog and won't reveal an ending
  unless you say spoilers are okay.

## Data

`data/movies.csv` and `data/reviews.csv` hold an original, fictional
30-title catalog spanning 2020–2026 with seed reviews, written for this
project so the app has real content to browse, filter, and chat about out
of the box. Everything (titles, cast, reviews) is invented — no real
filmography or reviews are represented.

## Notes

- Gemini model used: `gemini-2.5-flash` (Google's free tier as of testing).
  If Google rotates free-tier availability, swap the `GEMINI_MODEL`
  constant near the top of `app.py` for another free model, e.g.
  `gemini-3.1-flash-lite`.
- New reviews (including MovieBot-assisted ones you post) live in the
  session and reset when the app restarts — for permanent storage, swap
  `st.session_state.user_reviews` for a real database.
