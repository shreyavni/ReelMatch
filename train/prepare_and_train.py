"""
prepare_and_train.py
---------------------
Filters the raw TMDB+IMDB dataset down to a lightweight, deployable subset,
builds a content-based recommendation model (TF-IDF + cosine similarity),
and saves compact artifacts for the Flask backend to load at startup.

Run once, locally, before deployment:
    python prepare_and_train.py
"""

import pandas as pd
import numpy as np
import re
import pickle
import gzip
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

RAW_CSV = "/home/claude/dataset_extract/TMDB  IMDB Movies Dataset.csv"
OUT_DIR = "/home/claude/movie_recommender/backend/model_artifacts"
TOP_N_MOVIES = 12000          # keeps the final artifacts small & fast to load
MIN_VOTE_COUNT = 30           # drop obscure/low-signal entries
MAX_FEATURES = 15000          # cap TF-IDF vocabulary size

import os
os.makedirs(OUT_DIR, exist_ok=True)


def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x)


def build_soup(row):
    """Combine relevant text fields into one 'soup' for TF-IDF."""
    genres = clean_text(row["genres"]).replace(",", " ")
    keywords = clean_text(row["keywords"]).replace(",", " ")
    overview = clean_text(row["overview"])
    director = clean_text(row["directors"]).replace(",", " ")
    cast = clean_text(row["cast"]).replace(",", " ")
    # cast list can be long; keep first 5 to keep signal high & noise low
    cast_top5 = " ".join(cast.split(", ")[:5]) if cast else ""
    tagline = clean_text(row["tagline"])

    soup = f"{overview} {tagline} {genres} {genres} {keywords} {director} {director} {cast_top5}"
    soup = re.sub(r"[^a-zA-Z0-9\s]", " ", soup).lower()
    soup = re.sub(r"\s+", " ", soup).strip()
    return soup


def main():
    print("Loading raw CSV (this may take a minute)...")
    usecols = [
        "id", "title", "vote_average", "vote_count", "status", "release_date",
        "runtime", "adult", "original_language", "overview", "popularity",
        "poster_path", "tagline", "genres", "keywords", "directors", "cast",
    ]
    df = pd.read_csv(RAW_CSV, usecols=usecols, low_memory=False)
    print(f"Loaded {len(df)} rows")

    # --- Filtering to keep the deployed model lightweight & high quality ---
    df = df[df["status"] == "Released"]
    df = df[df["adult"] == False]  # noqa: E712
    df = df[df["poster_path"].notna()]
    df = df[df["overview"].notna() & (df["overview"].str.len() > 20)]
    df = df[df["vote_count"].fillna(0) >= MIN_VOTE_COUNT]
    df = df.drop_duplicates(subset=["title", "release_date"])

    # Rank by a blend of popularity and vote_count, take top N
    df["popularity"] = df["popularity"].fillna(0)
    df["vote_count"] = df["vote_count"].fillna(0)
    df["score_rank"] = df["popularity"].rank(pct=True) * 0.6 + df["vote_count"].rank(pct=True) * 0.4
    df = df.sort_values("score_rank", ascending=False).head(TOP_N_MOVIES)
    df = df.reset_index(drop=True)
    print(f"Kept {len(df)} movies after filtering")

    # --- Build text soup for content-based similarity ---
    print("Building text features...")
    df["soup"] = df.apply(build_soup, axis=1)

    # --- TF-IDF + Nearest Neighbors (memory-friendly: sparse matrix, no dense NxN matrix) ---
    print("Fitting TF-IDF vectorizer...")
    tfidf = TfidfVectorizer(max_features=MAX_FEATURES, stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["soup"])
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

    print("Fitting Nearest Neighbors index (cosine)...")
    nn_model = NearestNeighbors(n_neighbors=11, metric="cosine", algorithm="brute")
    nn_model.fit(tfidf_matrix)

    # --- Prepare lightweight metadata table for API responses ---
    df["release_year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year
    df["vote_average"] = df["vote_average"].fillna(0).round(1)
    df["poster_url"] = df["poster_path"].apply(
        lambda p: f"https://image.tmdb.org/t/p/w500{p}" if pd.notna(p) else None
    )

    meta = df[[
        "id", "title", "overview", "vote_average", "vote_count",
        "release_year", "poster_url", "genres", "runtime",
    ]].copy()
    meta["title_lower"] = meta["title"].str.lower().str.strip()

    # --- Save compact artifacts ---
    print("Saving artifacts...")
    with gzip.open(f"{OUT_DIR}/tfidf_matrix.pkl.gz", "wb") as f:
        pickle.dump(tfidf_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)

    with gzip.open(f"{OUT_DIR}/nn_model.pkl.gz", "wb") as f:
        pickle.dump(nn_model, f, protocol=pickle.HIGHEST_PROTOCOL)

    with gzip.open(f"{OUT_DIR}/tfidf_vectorizer.pkl.gz", "wb") as f:
        pickle.dump(tfidf, f, protocol=pickle.HIGHEST_PROTOCOL)

    meta.to_pickle(f"{OUT_DIR}/movies_meta.pkl.gz", compression="gzip")

    print("Done. Artifact sizes:")
    import subprocess
    subprocess.run(["du", "-sh", OUT_DIR])
    subprocess.run(["ls", "-la", OUT_DIR])


if __name__ == "__main__":
    main()
