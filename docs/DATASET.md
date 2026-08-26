# Dataset Details

## Source

**File:** `TMDB + IMDB Movies Dataset.csv` from Kaggle 
A merged export combining **TMDB** (The Movie Database) and **IMDB**
metadata for a large catalogue of films.

## Raw size

| Metric | Value |
|---|---|
| File size | ~268 MB (within the 50–500 MB requirement) |
| Rows | 441,028 movies |
| Columns | 29 |

## Columns used by this project

| Column | Used for |
|---|---|
| `title` | Display name, search/autocomplete, lookup key |
| `overview` | Description shown on each result card + TF-IDF text feature |
| `tagline` | Extra TF-IDF text feature |
| `genres` | Displayed as tags + TF-IDF feature |
| `keywords` | TF-IDF feature (themes/plot elements) |
| `directors` | TF-IDF feature |
| `cast` | Top 5 billed actors used as a TF-IDF feature |
| `vote_average` | Star rating shown on each card |
| `vote_count` | Used to rank/filter for quality and popularity |
| `popularity` | Used (with `vote_count`) to select the top movies kept in the model |
| `poster_path` | Combined with TMDB's image CDN to build `poster_url` |
| `release_date` | Parsed to `release_year` for display |
| `runtime` | Displayed as extra metadata |
| `status` | Filtered to `"Released"` only (drops unreleased/planned titles) |
| `adult` | Filtered to `False` only |

Columns such as `revenue`, `budget`, `homepage`, `production_companies`,
`production_countries`, `spoken_languages`, `writers`, `tconst`,
`backdrop_path`, and `id` are present in the raw file but **not** used by
the model or the API, to keep the artifact small.

## Cleaning & filtering steps (see `train/prepare_and_train.py`)

1. Keep only `status == "Released"` and `adult == False`.
2. Drop rows with no poster (`poster_path`) or no usable overview
   (< 20 characters, or missing).
3. Drop rows with `vote_count < 30` — removes very obscure/low-signal
   entries that would add noise without adding value to recommendations.
4. Drop duplicate `(title, release_date)` pairs.
5. Rank remaining rows by a blend of **popularity percentile (60%)** and
   **vote_count percentile (40%)**, and keep the **top 12,000 movies**.

This reduces the working set from 441k rows to 12,000 — small enough that
the resulting model artifacts total **~14 MB**, while still covering
essentially every movie a typical user is likely to search for (popular,
well-reviewed titles across decades and genres).

## Why filter this way instead of using the full 441k rows?

- **Deployment memory limits.** Free hosting tiers (Render/Railway/
  PythonAnywhere free plans) typically cap memory around 512 MB. A TF-IDF
  matrix and nearest-neighbor index built over 441k documents would be far
  too large and slow to fit comfortably.
- **Quality over quantity.** Many of the 441k rows are extremely obscure
  titles (a handful of votes, no real synopsis) that add noise to a
  content-based recommender without meaningfully improving results for
  real users.
- **The 50 MB dataset minimum applies to the source dataset, not the final
  deployed model** — the raw 268 MB CSV is used once, locally, to train
  the model; it is never uploaded to GitHub or the hosting platform (see
  `.gitignore`). Only the ~14 MB trained artifacts are deployed.
