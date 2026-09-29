# CineSense

CineSense is a Streamlit application for exploring movie metadata, MovieLens user ratings, and user-applied tags. Pandas prepares the data and visualizations; Google Gemini powers MovieBot using only a small, query-filtered subset of those records.

The bundled MovieLens snapshot does **not** contain written movie reviews. Users can optionally upload a legally reusable review CSV linked by MovieLens `movieId` or IMDb ID; this text stays in the current app session. CineSense does not claim MovieLens ratings are IMDb ratings and does not scrape IMDb.

## Project Overview

This CS 315 Activity 3 project demonstrates:

- Loading and validating related CSV data with Pandas.
- Interactive movie search, genre/year/rating/rating-count filters, sorting, pagination, and detail views.
- Rating distributions, genre comparisons, rating activity over time, and user-tag summaries.
- A Gemini chatbot grounded in locally filtered MovieLens records and aggregates.
- A data-driven similar-movie recommender based on genre overlap and rating confidence.
- Optional Wikidata/ Wikimedia Commons detail lookups and movie-linked review-text analysis.
- Session-only visitor reviews on movie detail pages; submissions display immediately with a separate VADER sentiment estimate.

No account or login is required.

## Dataset Source

CineSense bundles the **MovieLens latest-small** dataset from [GroupLens Research](https://grouplens.org/datasets/movielens/latest/). The archive was generated on **September 26, 2018** and contains rating activity through **September 24, 2018**. It is a development dataset, not a benchmark for shared research results.

The original snapshot describes 9,742 movies, 100,836 ratings, 3,683 tag applications, and 610 anonymized users. The app validates and reports the rows it actually loads. The source README is included at `data/MOVIELENS_README.txt` and is also [available online](https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html).

| File | Source fields | Use |
| --- | --- | --- |
| `data/movies.csv` | `movieId`, `title`, `genres` | MovieLens movie titles and pipe-separated genres |
| `data/ratings.csv` | `userId`, `movieId`, `rating`, `timestamp` | Anonymized user star-rating events, half-star increments from 0.5 to 5.0 |
| `data/tags.csv` | `userId`, `movieId`, `tag`, `timestamp` | Short user-applied tags; these are not reviews |
| `data/links.csv` | `movieId`, `imdbId`, `tmdbId` | Cross-reference identifiers supplied by MovieLens |

MovieLens documentation says some title metadata may have been entered manually or imported from TMDb. The IMDb IDs are links only. CineSense does not download IMDb ratings or scrape IMDb.

### License and Attribution

The bundled MovieLens data may be used and redistributed under the conditions in its source README. The conditions include attribution, no implied endorsement by the University of Minnesota or GroupLens, and no commercial or revenue-bearing use without permission. Review the [current dataset terms](https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html) before reuse or deployment. This project makes no endorsement claim.

Suggested citation: F. Maxwell Harper and Joseph A. Konstan (2015). “The MovieLens Datasets: History and Context.” *ACM Transactions on Interactive Intelligent Systems*, 5(4), 19:1–19:19. <https://doi.org/10.1145/2827872>.

## Data Preparation

`services/data_service.py` performs the preprocessing:

- Checks that every required file and column exists.
- Converts identifiers, ratings, and Unix timestamps to numeric/datetime values.
- Removes invalid ratings, malformed IDs/timestamps, duplicate movies, duplicate user/movie ratings, duplicate tag events, and references to unknown movies.
- Trims source text, extracts release years from title strings, and splits genres into normalized lists.
- Retains missing years as missing rather than guessing them.
- Calculates each title's mean observed rating and number of rating records.

Runtime, director, cast, synopsis, review text, and poster images are not in this dataset. Where MovieLens provides an IMDb cross-reference, a user can request optional Wikidata structured metadata. Fields remain explicitly unavailable when Wikidata lacks them. A Wikidata description is not presented as a full plot synopsis. Commons images link to their file page for individual license/attribution details.

## Ratings and Recommendation Method

“MovieLens average rating” is the mean of the actual MovieLens rating rows for a movie. It is not an IMDb rating. “Rating count” counts individual MovieLens rating events, not reviews.

“CineSense Weighted Score” is a Bayesian ranking used to keep movies with very few ratings from dominating the top of a list:

$$
WR = \frac{v}{v+m}R + \frac{m}{v+m}C
$$

Here $R$ is the movie's observed average, $v$ its rating count, $C$ the global MovieLens rating average, and $m=100$ the prior strength. This is a CineSense calculation, not a score published by IMDb or GroupLens.

Similar-movie suggestions first rank candidates by Jaccard overlap of their listed genres with the selected movie, then by CineSense Weighted Score and rating count. The detail view states this method.

## Application Pages

- **Home:** calculated dataset metrics, most-rated and weighted-score-ranked titles, popular genres, and a quick search.
- **Discover Movies:** literal title search, genre, year, minimum average rating, minimum rating count, six sort modes, pagination, and movie details.
- **Review Analytics:** rating distribution, rating-by-genre, rating activity by timestamp year, popular tags, and a clearly labeled rating-derived sentiment proxy. Upload a linked review CSV for VADER sentiment and optional review summaries.
- **Movie Details:** on-demand Wikidata director/cast/runtime/description/image lookup, a linked Wikipedia plot excerpt when available, MovieLens rating distribution, tags, similar titles, visitor-written reviews, and matching uploaded reviews.
- **MovieBot:** Gemini Q&A grounded in up to 12 filtered title records plus relevant small aggregates/tags.

Dataset provenance, preprocessing, calculation methods, AI methodology, and license notes are documented in this README rather than shown as a separate web-app page. The app explicitly distinguishes **source data**, **calculated values**, **optional Wikidata data**, **uploaded review data**, and **AI-generated answers**. MovieLens user tags are never represented as written reviews. The default sentiment chart is calculated from star-rating thresholds, not text.

### Optional Review Upload

In **Review Analytics**, provide the source/citation and upload a CSV with `review_text` and either `movieId` or `imdbId`. The header-only [review upload template](data/review_upload_template.csv) shows the supported columns. An optional `sentiment` column accepts positive/neutral/negative source labels. Rows without a recognized source label receive a VADER polarity estimate. Uploads are limited to 10 MB, 20,000 rows, and 10,000 characters per review; unmatched movie IDs are rejected. Data remains in Streamlit session memory and is not written to project files. Only upload text you have permission to use. The Stanford ACL IMDb sentiment corpus does not contain movie identifiers, so this project does not map its reviews onto individual titles.

### Visitor Reviews

Anyone can post a review from a movie's **Write a review** form without creating an account. A submission includes an optional display name, a 1–5 star rating, and up to 2,000 characters of text. It appears on that movie's detail page immediately and receives a VADER sentiment estimate. Visitor reviews are labeled separately from MovieLens and imported review datasets; they are held in Streamlit session memory and are not permanent or shared between users. A database would be required for persistence.

### Optional Metadata Enrichment

From a movie's details, **Load Wikidata details** first queries the Wikidata SPARQL endpoint using the MovieLens IMDb cross-reference. If SPARQL is unavailable or returns no match, CineSense searches Wikipedia and Wikidata APIs, then verifies that the linked Wikidata item claims the same IMDb ID before using its fields. Results may include director, cast, runtime, a short Wikidata description, and a Wikimedia Commons image. When Wikidata links to an English Wikipedia article with a Plot section, CineSense can show a bounded excerpt with article attribution and a CC BY-SA 4.0 license link. Wikidata structured data is CC0; Commons images link to their file page for individual license/attribution details. Lookups are on demand; coverage is incomplete and the endpoints can rate-limit requests. Descriptions are not represented as full plot summaries.

## GenAI Integration

MovieBot parses supported genre, year, minimum-rating, title, popularity, and similar-title cues before constructing a compact context. It does not send the full dataset to Gemini. Its system instructions prohibit presenting tags as reviews, MovieLens ratings as IMDb ratings, or unavailable metadata as fact. If a question needs absent fields, the assistant is instructed to say that the information is unavailable.

AI answers can still be mistaken. The app provides source counts and charts so results can be checked. Without a Gemini key, browsing, filters, recommendations, and analytics remain available.

## Installation and Running Locally

Python 3.10 or newer is recommended.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

On macOS/Linux, activate with `source .venv/bin/activate` instead. Streamlit prints the local URL, normally `http://localhost:8501`.

## Gemini Secret

Create `.streamlit/secrets.toml` (do not commit it) with:

```toml
GEMINI_API_KEY = "your-key-here"
```

Get a key from [Google AI Studio](https://aistudio.google.com/apikey). For deployment, enter the same setting under the Streamlit Community Cloud app's **Settings → Secrets**. The server reads Streamlit secrets first and then the `GEMINI_API_KEY` environment variable. Never expose the key in the UI or source control.

## Streamlit Community Cloud Deployment

1. Push the project to a GitHub repository without `.streamlit/secrets.toml`.
2. Create a Streamlit Community Cloud app using `app.py` as the entry point.
3. Add `GEMINI_API_KEY` in the app's Secrets settings.
4. Deploy and verify dataset loading, analytics, and a Gemini response.

Deployment and redistribution must remain consistent with the MovieLens license terms; this dataset is not permitted for commercial or revenue-bearing use without prior permission.

## Project Structure

```text
cinesense_app/
├── app.py
├── requirements.txt
├── README.md
├── services/
│   ├── analytics_service.py
│   ├── data_service.py
│   ├── metadata_service.py
│   ├── moviebot_service.py
│   └── review_service.py
├── data/
│   ├── movies.csv
│   ├── ratings.csv
│   ├── tags.csv
│   ├── links.csv
│   ├── review_upload_template.csv
│   └── MOVIELENS_README.txt
└── .streamlit/
    └── config.toml
```

## Testing

Run the offline test suite (no Gemini key or network call is needed):

```powershell
python -m unittest discover -s tests -v
```

The tests cover data counts and cleaning, analytics totals, genre-based recommendations, query filters, bounded MovieBot context, ten example question shapes, review-file validation and joins, and explicit handling of absent fields. Then run `streamlit run app.py` and verify the desktop navigation, filters, details, charts, upload flow, missing-key state, and a Gemini request if a key is configured.

## Limitations and Future Improvements

- The snapshot is from 2018 and reflects MovieLens users, not the general population.
- User tags are not review prose and cannot support sentiment or opinion analysis.
- MovieLens does not provide plot, cast, director, runtime, poster, or written-review data in this release; Wikidata enrichment is optional and sparse.
- Review-text analytics require the user to provide a compatible dataset and source/reuse information.
- MovieBot requires internet access and a configured Gemini key; generative answers are not guaranteed correct.
- The MovieLens latest-small release is a development dataset that may change if downloaded again; this project bundles the dated archive snapshot for reproducibility.
- Future work could add a separately licensed review-text dataset with its own clearly labeled analysis pipeline, or a newer metadata source after its licensing and deployment terms are reviewed.