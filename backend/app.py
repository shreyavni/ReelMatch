"""
app.py - Movie Recommendation API (Flask)

Endpoints:
    GET  /health              -> {"status": "OK"}
    POST /recommend            -> body: {"movie": "<title>"} -> top 10 similar movies
    GET  /movies?q=<query>     -> lightweight search/autocomplete for the frontend

The trained model (TF-IDF + Nearest Neighbors) is loaded ONCE at server
startup, not per-request, to keep response times fast and memory stable.
"""

import os
import gzip
import pickle
import traceback

import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# --------------------------------------------------------------------------
# App setup
# --------------------------------------------------------------------------
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="")
CORS(app)  # allow the frontend to call the API even if hosted on a different origin

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_artifacts")

# --------------------------------------------------------------------------
# Load model artifacts ONCE at startup
# --------------------------------------------------------------------------
_model_load_error = None
tfidf_matrix = None
nn_model = None
tfidf_vectorizer = None
movies_meta = None

try:
    print("Loading model artifacts...")
    with gzip.open(os.path.join(ARTIFACT_DIR, "tfidf_matrix.pkl.gz"), "rb") as f:
        tfidf_matrix = pickle.load(f)

    with gzip.open(os.path.join(ARTIFACT_DIR, "nn_model.pkl.gz"), "rb") as f:
        nn_model = pickle.load(f)

    with gzip.open(os.path.join(ARTIFACT_DIR, "tfidf_vectorizer.pkl.gz"), "rb") as f:
        tfidf_vectorizer = pickle.load(f)

    movies_meta = pd.read_pickle(os.path.join(ARTIFACT_DIR, "movies_meta.pkl.gz"), compression="gzip")
    movies_meta = movies_meta.reset_index(drop=True)

    # Build a fast lookup: lowercase title -> row index
    TITLE_TO_INDEX = {t: i for i, t in enumerate(movies_meta["title_lower"])}

    print(f"Model loaded successfully. {len(movies_meta)} movies available.")
except Exception as e:  # noqa: BLE001
    _model_load_error = str(e)
    TITLE_TO_INDEX = {}
    print("ERROR loading model artifacts:", e)
    traceback.print_exc()


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def movie_row_to_dict(idx, score=None):
    row = movies_meta.iloc[idx]
    result = {
        "title": row["title"],
        "overview": row["overview"] if isinstance(row["overview"], str) else "",
        "rating": float(row["vote_average"]) if pd.notna(row["vote_average"]) else None,
        "vote_count": int(row["vote_count"]) if pd.notna(row["vote_count"]) else None,
        "release_year": int(row["release_year"]) if pd.notna(row["release_year"]) else None,
        "poster_url": row["poster_url"] if isinstance(row["poster_url"], str) else None,
        "genres": row["genres"] if isinstance(row["genres"], str) else "",
        "runtime": int(row["runtime"]) if pd.notna(row["runtime"]) else None,
    }
    if score is not None:
        result["similarity"] = round(float(score), 4)
    return result


def find_closest_title(query):
    """Find the best matching known title for a (possibly imperfect) user query."""
    query_lower = query.strip().lower()
    if query_lower in TITLE_TO_INDEX:
        return query_lower

    # substring match fallback
    candidates = [t for t in TITLE_TO_INDEX if query_lower in t]
    if candidates:
        # prefer the shortest matching title (closest to exact match)
        candidates.sort(key=len)
        return candidates[0]

    return None


# --------------------------------------------------------------------------
# Frontend
# --------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return send_from_directory(STATIC_DIR, "index.html")


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    if _model_load_error:
        return jsonify({"status": "ERROR", "detail": _model_load_error}), 500
    return jsonify({"status": "OK", "movies_loaded": len(movies_meta)}), 200


@app.route("/movies", methods=["GET"])
def search_movies():
    """Lightweight search for frontend autocomplete: /movies?q=incep"""
    if _model_load_error:
        return jsonify({"error": "Model not loaded on server."}), 500

    query = request.args.get("q", "").strip().lower()
    if not query:
        return jsonify({"results": []}), 200

    matches = movies_meta[movies_meta["title_lower"].str.contains(query, na=False, regex=False)]
    matches = matches.sort_values("vote_count", ascending=False).head(8)
    results = [{"title": r["title"], "release_year": (int(r["release_year"]) if pd.notna(r["release_year"]) else None)}
               for _, r in matches.iterrows()]
    return jsonify({"results": results}), 200


@app.route("/recommend", methods=["POST"])
def recommend():
    """
    Body (JSON): {"movie": "Inception"}
    Returns: top 10 movies similar in content to the given title.
    """
    if _model_load_error:
        return jsonify({"error": "Model failed to load on the server. Try again later."}), 500

    try:
        data = request.get_json(silent=True)
    except Exception:  # noqa: BLE001
        data = None

    if not data or "movie" not in data:
        return jsonify({"error": "Missing 'movie' field in request body. Example: {\"movie\": \"Inception\"}"}), 400

    movie_query = data.get("movie", "")
    if not isinstance(movie_query, str) or not movie_query.strip():
        return jsonify({"error": "'movie' must be a non-empty string."}), 400

    try:
        n_results = int(data.get("count", 10))
        n_results = max(1, min(n_results, 20))
    except (TypeError, ValueError):
        n_results = 10

    matched_title = find_closest_title(movie_query)
    if matched_title is None:
        return jsonify({
            "error": f"No movie found matching '{movie_query}'. "
                     f"Try a different spelling or check the /movies?q= search endpoint.",
        }), 404

    try:
        idx = TITLE_TO_INDEX[matched_title]
        distances, indices = nn_model.kneighbors(
            tfidf_matrix[idx], n_neighbors=min(n_results + 1, tfidf_matrix.shape[0])
        )

        recs = []
        for dist, rec_idx in zip(distances[0], indices[0]):
            if rec_idx == idx:
                continue  # skip the movie itself
            similarity = 1 - dist  # cosine distance -> similarity
            recs.append(movie_row_to_dict(rec_idx, score=similarity))
            if len(recs) >= n_results:
                break

        return jsonify({
            "query": movie_query,
            "matched_title": movies_meta.iloc[idx]["title"],
            "recommendations": recs,
        }), 200

    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return jsonify({"error": "Something went wrong while generating recommendations.", "detail": str(e)}), 500


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found."}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed for this endpoint."}), 405


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error."}), 500


# --------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
