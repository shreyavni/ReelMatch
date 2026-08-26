# API Testing Documentation (Postman)

All endpoints below were tested locally with Postman before deployment.
Base URL shown as `http://127.0.0.1:5000` — replace with your deployed URL
(e.g. `https://reel-match.onrender.com`) once live.

---

## 1. `GET /health`

| Field | Value |
|---|---|
| **Endpoint URL** | `http://127.0.0.1:5000/health` |
| **Request method** | GET |
| **Input sent** | none |
| **Response received** | `{"status": "OK", "movies_loaded": 12000}` |
| **Response time** | ~15 ms |
| **Status code** | 200 |

**Error case** — if the model fails to load on the server:
`{"status": "ERROR", "detail": "<reason>"}` — Status code `500`.

---

## 2. `GET /movies`

Lightweight search used for the frontend's autocomplete dropdown.

| Field | Value |
|---|---|
| **Endpoint URL** | `http://127.0.0.1:5000/movies?q=incep` |
| **Request method** | GET |
| **Input sent** | Query param `q=incep` |
| **Response received** | `{"results": [{"title": "Inception", "release_year": 2010}]}` |
| **Response time** | ~20 ms |
| **Status code** | 200 |

**Error case** — empty/missing query:
`GET /movies` (no `q`) → `{"results": []}` — Status code `200` (not an error, just empty).

---

## 3. `POST /recommend`

Main recommendation endpoint.

| Field | Value |
|---|---|
| **Endpoint URL** | `http://127.0.0.1:5000/recommend` |
| **Request method** | POST |
| **Headers** | `Content-Type: application/json` |
| **Input sent** | `{"movie": "Inception"}` |
| **Response received** | `{"query": "Inception", "matched_title": "Inception", "recommendations": [ {...10 movie objects...} ]}` |
| **Response time** | ~50 ms |
| **Status code** | 200 |

### Error case 1 — missing `movie` field
- **Input sent:** `{}`
- **Response received:** `{"error": "Missing 'movie' field in request body. Example: {\"movie\": \"Inception\"}"}`
- **Status code:** 400

### Error case 2 — empty / malformed JSON body
- **Input sent:** *(empty body)*
- **Response received:** `{"error": "Missing 'movie' field in request body. Example: {\"movie\": \"Inception\"}"}`
- **Status code:** 400
- Server does **not** crash or return a raw 500 page — confirmed by `traceback`-free response.

### Error case 3 — movie not found in dataset
- **Input sent:** `{"movie": "asdkjaskldjaklsjd"}`
- **Response received:** `{"error": "No movie found matching 'asdkjaskldjaklsjd'. Try a different spelling or check the /movies?q= search endpoint."}`
- **Status code:** 404

### Error case 4 — unsupported HTTP method
- **Input sent:** `GET /recommend`
- **Response received:** `{"error": "Method not allowed for this endpoint."}`
- **Status code:** 405

---

## 4. Unknown route (`GET /doesnotexist`)

| Field | Value |
|---|---|
| **Response received** | `{"error": "Endpoint not found."}` |
| **Status code** | 404 |

---

## Postman Collection

Import `Reel_Match.postman_collection.json` (included in this `docs/` folder)
into Postman to run all the requests above with one click.
