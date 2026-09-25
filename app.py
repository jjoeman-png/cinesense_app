"""
CineSense — Discover, Rate, and Review Movies
Streamlit app · Google Gemini (GenAI) powers the MovieBot assistant.
"""

import os
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st
from google import genai
from google.genai import types

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(page_title="CineSense", page_icon="🎬", layout="wide")

DATA_DIR = Path(__file__).parent / "data"
GEMINI_MODEL = "gemini-3.1-flash-lite"

GENRE_STYLE = {
    "Drama":        ("🎭", "#8b5cf6"),
    "Action":       ("💥", "#ef4444"),
    "Thriller":     ("🔎", "#0ea5e9"),
    "Sci-Fi":       ("🚀", "#22c55e"),
    "Comedy":       ("😂", "#f59e0b"),
    "Romance":      ("💕", "#ec4899"),
    "Horror":       ("👻", "#6b21a8"),
    "Documentary":  ("🎥", "#64748b"),
    "Animation":    ("🎨", "#06b6d4"),
    "Fantasy":      ("🗺️", "#a855f7"),
}

# --------------------------------------------------------------------------
# Theme — dark, IMDb-inspired
# --------------------------------------------------------------------------
st.markdown("""
<style>
:root{
  --bg:#0e0e10; --panel:#1a1a1e; --panel2:#212126; --ink:#f2f2f3; --sub:#9a9aa2;
  --gold:#f5c518; --border:#2c2c33;
}
html, body, [class*="css"]{ background:var(--bg); color:var(--ink); }
.stApp{ background:var(--bg); }
h1,h2,h3,h4{ color:var(--ink); }
.brand{ font-size:2.1rem; font-weight:800; letter-spacing:-.02em; }
.brand span{ color:var(--gold); }
.tagline{ color:var(--sub); margin-top:-8px; }
.movie-card{
  background:var(--panel); border:1px solid var(--border); border-radius:12px;
  overflow:hidden; margin-bottom:16px; height:100%;
}
.poster{
  height:140px; display:flex; align-items:center; justify-content:center;
  font-size:2.6rem;
}
.card-body{ padding:10px 12px 14px; }
.card-title{ font-weight:700; font-size:0.98rem; margin:0 0 2px; color:var(--ink); }
.card-meta{ color:var(--sub); font-size:0.8rem; margin-bottom:6px; }
.stars{ color:var(--gold); font-size:0.95rem; letter-spacing:1px; }
.pill{
  display:inline-block; background:var(--panel2); border:1px solid var(--border);
  color:var(--sub); border-radius:999px; padding:2px 10px; font-size:0.75rem; margin-right:6px;
}
.review-box{
  background:var(--panel); border:1px solid var(--border); border-radius:10px;
  padding:12px 14px; margin-bottom:10px;
}
.review-head{ display:flex; justify-content:space-between; font-size:0.85rem; color:var(--sub); }
section[data-testid="stSidebar"]{ background:var(--panel); }
div[data-testid="stTextInput"] input, div[data-testid="stTextArea"] textarea{
  background:var(--panel2); color:var(--ink); border-radius:8px;
}
.st-key-moviebot_float{
  position:fixed; bottom:22px; right:22px; z-index:9999;
}
.st-key-moviebot_float button{
  border-radius:999px !important; font-weight:700 !important;
  background:var(--gold) !important; color:#111 !important; border:none !important;
  box-shadow:0 4px 14px rgba(0,0,0,.4);
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Data loading & cleaning
# --------------------------------------------------------------------------
@st.cache_data
def load_movies() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "movies.csv")
    for col in ["title", "genre", "director", "cast", "synopsis"]:
        df[col] = df[col].astype(str).str.strip()
    df["plot_spoiler"] = df["plot_spoiler"].fillna("").astype(str).str.strip()
    df = df.dropna(subset=["title"]).drop_duplicates(subset=["id"]).reset_index(drop=True)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["runtime"] = pd.to_numeric(df["runtime"], errors="coerce").astype("Int64")
    return df


@st.cache_data
def load_base_reviews() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "reviews.csv")
    df["reviewer"] = df["reviewer"].astype(str).str.strip()
    df["review_text"] = df["review_text"].astype(str).str.strip()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["review_text", "rating"])
    return df


movies_df = load_movies()
base_reviews_df = load_base_reviews()

if "user_reviews" not in st.session_state:
    st.session_state.user_reviews = []
if "selected_movie_id" not in st.session_state:
    st.session_state.selected_movie_id = None
if "movie_chat" not in st.session_state:
    st.session_state.movie_chat = []


def all_reviews_for(movie_id: int) -> pd.DataFrame:
    extra = pd.DataFrame([r for r in st.session_state.user_reviews if r["movie_id"] == movie_id])
    base = base_reviews_df[base_reviews_df["movie_id"] == movie_id]
    combined = pd.concat([base, extra], ignore_index=True) if not extra.empty else base
    return combined.sort_values("review_date", ascending=False)


def rating_summary(movie_id: int):
    revs = all_reviews_for(movie_id)
    if revs.empty:
        return None, 0
    return round(revs["rating"].mean(), 1), len(revs)


def stars_html(rating) -> str:
    if rating is None:
        return '<span class="card-meta">Not yet rated</span>'
    full = int(round(rating))
    return f'<span class="stars">{"★" * full}{"☆" * (5 - full)}</span> <span class="card-meta">{rating}/5</span>'


movies_df["avg_rating"] = movies_df["id"].apply(lambda i: rating_summary(i)[0])
movies_df["review_count"] = movies_df["id"].apply(lambda i: rating_summary(i)[1])


# --------------------------------------------------------------------------
# GenAI client (server-side key — end users never enter an API key)
# --------------------------------------------------------------------------
def _secret(name: str) -> str:
    # st.secrets raises if no secrets.toml exists at all (e.g. before the
    # site owner has configured one), so guard the lookup instead of using
    # st.secrets.get(), which does not catch that case.
    try:
        return st.secrets[name]
    except Exception:
        return ""


@st.cache_resource
def get_client():
    key = _secret("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY", "")
    return genai.Client(api_key=key) if key else None


def build_catalog_context() -> str:
    lines = []
    for row in movies_df.itertuples():
        rating_txt = f"{row.avg_rating}/5 ({row.review_count} reviews)" if row.avg_rating else "not yet rated"
        lines.append(
            f"[{row.id}] \"{row.title}\" ({row.year}, {row.genre}) — dir. {row.director}; "
            f"starring {row.cast}; {row.runtime} min; rating {rating_txt}. "
            f"Synopsis: {row.synopsis} Ending (spoiler, only share if asked): {row.plot_spoiler or 'n/a'}"
        )
    return "\n".join(lines)


def ask_moviebot(user_message: str) -> str:
    client = get_client()
    if client is None:
        return "MovieBot is temporarily unavailable right now — please try again later."

    system_instruction = (
        "You are MovieBot, the friendly in-app assistant for CineSense, a movie discovery and review site. "
        "You have exactly five jobs: (1) Movie search — help users find movies in the catalog below by title, "
        "actor, director, or genre. (2) Movie recommendations — suggest movies from the catalog based on mood, "
        "genre, or similarity to a title the user liked. (3) Movie information — answer questions about cast, "
        "director, runtime, synopsis, or rating. (4) Review-writing assistance — help users turn their raw "
        "thoughts into a short, polished review they can post. (5) Review summarization — summarize the general "
        "consensus of a movie's reviews when asked. "
        "Only discuss movies that appear in the catalog below; if something isn't in the catalog, say so "
        "honestly instead of making it up. Never reveal a plot ending unless the user explicitly says spoilers "
        "are okay or directly asks 'what happens' / 'explain the ending'. Keep replies short, warm, and a little "
        "playful — a sentence or two for simple questions, a short list for recommendations. Use at most one "
        "relevant emoji per reply.\n\nCATALOG:\n" + build_catalog_context()
    )

    history = st.session_state.movie_chat[-8:]
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in history)
    prompt = f"{convo}\nuser: {user_message}\nassistant:"

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(system_instruction=system_instruction, temperature=0.5),
        )
        return response.text or "Hmm, I didn't catch that — could you try rephrasing?"
    except Exception:
        return "Something went wrong reaching MovieBot. Please try again in a moment."


def send_to_moviebot(text: str):
    st.session_state.movie_chat.append({"role": "user", "content": text})
    reply = ask_moviebot(text)
    st.session_state.movie_chat.append({"role": "assistant", "content": reply})


# --------------------------------------------------------------------------
# Reusable UI pieces
# --------------------------------------------------------------------------
def poster(genre: str):
    emoji, color = GENRE_STYLE.get(genre, ("🎬", "#374151"))
    return f'<div class="poster" style="background:linear-gradient(135deg,{color}55,{color}22);">{emoji}</div>'


def movie_card(row, section: str):
    with st.container(border=False):
        st.markdown(
            f"""<div class="movie-card">
                  {poster(row.genre)}
                  <div class="card-body">
                    <p class="card-title">{row.title}</p>
                    <p class="card-meta">{row.year} · {row.genre} · {row.runtime} min</p>
                    {stars_html(row.avg_rating)}
                  </div>
                </div>""",
            unsafe_allow_html=True,
        )
        if st.button("View details", key=f"view_{section}_{row.id}", use_container_width=True):
            st.session_state.selected_movie_id = row.id
            st.rerun()


def movie_grid(df: pd.DataFrame, section: str, per_row: int = 4):
    if df.empty:
        st.info("No movies match your filters.")
        return
    rows = [df.iloc[i:i + per_row] for i in range(0, len(df), per_row)]
    for chunk in rows:
        cols = st.columns(per_row)
        for col, (_, row) in zip(cols, chunk.iterrows()):
            with col:
                movie_card(row, section)


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown('<p class="brand">🎬 Cine<span>Sense</span></p>', unsafe_allow_html=True)
st.markdown('<p class="tagline">Discover, rate, and review movies — 2020 to 2026.</p>', unsafe_allow_html=True)
st.write("")

# --------------------------------------------------------------------------
# Sidebar — browse & filter
# --------------------------------------------------------------------------
with st.sidebar:
    st.subheader("🔎 Search & Filter")
    search_query = st.text_input("Search title, actor, or director", placeholder="e.g. Devon Cross")
    genre_filter = st.multiselect("Genre", sorted(movies_df["genre"].unique()), default=None)
    year_filter = st.slider("Year", 2020, 2026, (2020, 2026))
    sort_by = st.selectbox("Sort by", ["Top rated", "Newest", "Title A-Z"])

filtered = movies_df[movies_df["year"].between(year_filter[0], year_filter[1])]
if genre_filter:
    filtered = filtered[filtered["genre"].isin(genre_filter)]
if search_query:
    q = search_query.lower()
    filtered = filtered[
        filtered["title"].str.lower().str.contains(q)
        | filtered["cast"].str.lower().str.contains(q)
        | filtered["director"].str.lower().str.contains(q)
    ]
if sort_by == "Top rated":
    filtered = filtered.sort_values(["avg_rating", "review_count"], ascending=False, na_position="last")
elif sort_by == "Newest":
    filtered = filtered.sort_values("year", ascending=False)
else:
    filtered = filtered.sort_values("title")

# --------------------------------------------------------------------------
# Main content — detail view OR browse view
# --------------------------------------------------------------------------
if st.session_state.selected_movie_id is not None:
    mid = st.session_state.selected_movie_id
    row = movies_df[movies_df["id"] == mid]
    if row.empty:
        st.session_state.selected_movie_id = None
        st.rerun()
    m = row.iloc[0]

    if st.button("← Back to browsing"):
        st.session_state.selected_movie_id = None
        st.rerun()

    c1, c2 = st.columns([1, 2.2])
    with c1:
        st.markdown(poster(m.genre), unsafe_allow_html=True)
        st.markdown(f'<div style="height:14px"></div>{stars_html(m.avg_rating)}', unsafe_allow_html=True)
    with c2:
        st.markdown(f"## {m.title}")
        st.markdown(f'<span class="pill">{m.year}</span><span class="pill">{m.genre}</span><span class="pill">{m.runtime} min</span>', unsafe_allow_html=True)
        st.write("")
        st.write(f"**Director:** {m.director}")
        st.write(f"**Starring:** {m.cast}")
        st.write(f"**Synopsis:** {m.synopsis}")

    st.divider()
    st.subheader(f"Reviews ({m.review_count})")
    reviews = all_reviews_for(mid)
    if reviews.empty:
        st.caption("No reviews yet — be the first to write one below.")
    else:
        for r in reviews.itertuples():
            st.markdown(
                f"""<div class="review-box">
                      <div class="review-head"><b style="color:var(--ink)">{r.reviewer}</b><span>{r.review_date}</span></div>
                      <div class="stars">{"★" * int(r.rating)}{"☆" * (5 - int(r.rating))}</div>
                      <div style="margin-top:4px;">{r.review_text}</div>
                    </div>""",
                unsafe_allow_html=True,
            )

    with st.expander("✍️ Write a review"):
        with st.form(f"review_form_{mid}", clear_on_submit=True):
            reviewer_name = st.text_input("Your name", value="")
            star_rating = st.select_slider("Your rating", options=[1, 2, 3, 4, 5], value=4)
            review_text = st.text_area("Your review", placeholder="What did you think?")
            submitted = st.form_submit_button("Post review", type="primary")
            if submitted:
                if not review_text.strip():
                    st.warning("Please write something before posting.")
                else:
                    st.session_state.user_reviews.append({
                        "review_id": 10_000 + len(st.session_state.user_reviews),
                        "movie_id": mid,
                        "reviewer": reviewer_name.strip() or "Anonymous",
                        "rating": star_rating,
                        "review_text": review_text.strip(),
                        "review_date": date.today().isoformat(),
                    })
                    st.success("Review posted!")
                    st.rerun()

else:
    tab_home, tab_browse = st.tabs(["🏠 Home", "🔍 Browse all"])

    with tab_home:
        trending = movies_df[(movies_df["year"] >= 2025) | (movies_df["review_count"] >= 3)]
        trending = trending.sort_values(["avg_rating", "review_count"], ascending=False, na_position="last").head(8)
        st.markdown("### 🔥 Trending now")
        movie_grid(trending, section="trending")

        top_rated = movies_df.dropna(subset=["avg_rating"]).sort_values("avg_rating", ascending=False).head(8)
        st.markdown("### ⭐ Top rated")
        movie_grid(top_rated, section="top")

    with tab_browse:
        st.markdown(f"### {len(filtered)} movies")
        movie_grid(filtered, section="browse")


# --------------------------------------------------------------------------
# MovieBot — floating chat assistant
# --------------------------------------------------------------------------
bot_available = get_client() is not None

with st.container(key="moviebot_float"):
    with st.popover("💬 MovieBot"):
        st.markdown("**Ask MovieBot**")
        if not bot_available:
            st.caption("MovieBot is temporarily unavailable right now.")
        else:
            suggestions = [
                "🎬 Recommend me a horror movie",
                "⭐ What's the rating of Iron Meridian?",
                "📝 Help me review a movie I just watched",
                "🔥 What's trending?",
                "⚠️ Explain the ending of Silent Ledger, spoilers ok",
            ]
            for s in suggestions:
                if st.button(s, key=f"chip_{s}", use_container_width=True):
                    send_to_moviebot(s.split(" ", 1)[1])

            st.divider()
            for m in st.session_state.movie_chat[-10:]:
                with st.chat_message(m["role"]):
                    st.write(m["content"])

            with st.form("moviebot_form", clear_on_submit=True):
                user_msg = st.text_input("Message", label_visibility="collapsed", placeholder="Ask about any movie...")
                sent = st.form_submit_button("Send", type="primary")
                if sent and user_msg.strip():
                    send_to_moviebot(user_msg.strip())
                    st.rerun()
