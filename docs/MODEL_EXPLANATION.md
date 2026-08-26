# Model Explanation — How the Recommender Works

## Type of model

This is a **content-based recommendation system**, built with classic NLP
and machine learning — no deep learning, no GPU, no external AI API calls
at request time. It was chosen because it is lightweight (fits free hosting
memory limits), fast (~50 ms per request), fully explainable, and doesn't
need a large user-interaction dataset (which this project doesn't have) —
unlike collaborative filtering, which needs user ratings history.

## Step by step

### 1. Build a "soup" of text per movie

For every movie kept after filtering (see `docs/DATASET.md`), we combine
several fields into one text blob:

```
overview + tagline + genres (x2 for extra weight) + keywords + director (x2) + top-5 cast
```

Genres and director are repeated to give them slightly more weight than a
single mention of a keyword, since two movies sharing a genre and director
are usually a stronger content match than two movies sharing one obscure
keyword.

### 2. TF-IDF vectorization

Each movie's text soup is converted into a **TF-IDF (Term Frequency –
Inverse Document Frequency)** vector — a standard NLP technique that turns
text into numbers, where:
- Words that appear **often in a specific movie's description** but
  **rarely across the whole dataset** get a high weight (e.g. "wormhole",
  "heist").
- Common words that appear everywhere (e.g. "the", "story", stop words)
  get a low weight and are effectively ignored.

This produces one sparse numeric vector per movie (vocabulary capped at
15,000 terms to control memory).

### 3. Nearest Neighbors search (cosine similarity)

To find movies similar to a given title, we use **scikit-learn's
`NearestNeighbors`** with **cosine similarity** as the distance metric.
Cosine similarity measures the *angle* between two movies' TF-IDF vectors —
it's high when two movies use similar language/genres/cast/director
regardless of how long their descriptions are.

When a user asks for recommendations based on "Inception":
1. Look up Inception's TF-IDF vector.
2. Find the 10 nearest vectors (by cosine similarity) among the other
   11,999 movies.
3. Return those 10 movies, sorted by similarity score (shown as a
   "% match" badge in the UI).

### 4. Why Nearest Neighbors instead of a full similarity matrix?

A naive approach would precompute a full N×N cosine-similarity matrix
(12,000 × 12,000 ≈ 144 million values) — too much memory for free hosting.
`NearestNeighbors` instead computes similarity **on demand**, only for the
one movie being queried, against the sparse TF-IDF matrix. This keeps
memory flat regardless of catalogue size and is still fast (~50 ms) because
the matrix is sparse.

## What's saved to disk (`backend/model_artifacts/`)

| File | Contents |
|---|---|
| `tfidf_vectorizer.pkl.gz` | The fitted TF-IDF vectorizer (vocabulary + IDF weights) |
| `tfidf_matrix.pkl.gz` | The sparse TF-IDF matrix for all 12,000 movies |
| `nn_model.pkl.gz` | The fitted Nearest Neighbors index |
| `movies_meta.pkl.gz` | Lightweight metadata table (title, overview, rating, poster URL, etc.) used to build API responses |

All four are gzip-compressed pickles, ~14 MB combined, and are **loaded
once when the Flask server starts** (see `backend/app.py`), not on every
request — so repeated requests only pay the cost of the nearest-neighbor
lookup, not the cost of reloading the model.

## Limitations (transparently noted)

- **Cold-start for brand-new/unlisted movies:** if a user searches for a
  title not in the top-12,000 subset, the API returns a clear 404 error and
  suggests using `/movies?q=` to find a close match — it does not crash or
  guess randomly.
- **Content-based only:** recommendations are based on what a movie is
  *about* (genre, plot, cast, director), not on what other users with
  similar taste watched (that would be collaborative filtering, which needs
  user-rating history this dataset doesn't provide).
