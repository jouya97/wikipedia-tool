"""Wikipedia search client.

Implements the locked MediaWiki Action API single-call pattern from
``PLANS.md`` (generator=search chained into prop=extracts) and exposes
``search_wikipedia(query: str) -> dict`` per the pre-spawn contract.

Contract (locked in ``PLANS.md``):

    Return shape:
        {"title": str | None, "url": str | None, "lead": str, "error": str | None}

    - Success: title/url populated, lead is plaintext (no wiki markup),
      error is None.
    - No-hit / failure: title=None, url=None, lead="", error="<reason>".
      Never raises.
    - Cache: SQLite at ``cache/wikipedia.sqlite`` keyed on (endpoint, query).
      Forward-compatible schema (extra columns can be added later without
      breaking existing rows).
    - Backoff: exponential + jitter on HTTP 429 / 5xx.
    - User-Agent: per Wikimedia API etiquette.

This module is shared between ``agent/haiku_agent.py`` and the data-gen
workers — keep it dumb and predictable.
"""

from __future__ import annotations

import json
import os
import random
import sqlite3
import time
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import urlencode

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WIKIPEDIA_ENDPOINT = "https://en.wikipedia.org/w/api.php"

# Per Wikimedia API etiquette — see PLANS.md "Decisions" section.
USER_AGENT = (
    "wikipedia-tool/0.1 "
    "(https://github.com/jouyang97/wikipedia-tool; jianouyang001@gmail.com)"
)

# Backoff knobs. Total worst-case wait ~ sum(0.5 * 2**i for i in 0..4) ≈ 15.5s
# plus jitter. Tunable via env vars for tests.
_MAX_RETRIES = int(os.environ.get("WIKI_CLIENT_MAX_RETRIES", "5"))
_BASE_BACKOFF_S = float(os.environ.get("WIKI_CLIENT_BASE_BACKOFF_S", "0.5"))
_REQUEST_TIMEOUT_S = float(os.environ.get("WIKI_CLIENT_TIMEOUT_S", "20"))

# Cache file. Test code overrides via ``set_cache_path`` to point at a tmp
# location so the production cache is never touched.
_DEFAULT_CACHE_PATH = (
    Path(__file__).resolve().parent.parent / "cache" / "wikipedia.sqlite"
)
_cache_path: Path = _DEFAULT_CACHE_PATH
_cache_init_lock = Lock()
_cache_initialized_for: Path | None = None


# ---------------------------------------------------------------------------
# Cache (SQLite)
# ---------------------------------------------------------------------------

# Forward-compatible schema:
#   - composite PK on (endpoint, query) is the canonical key
#   - response_json holds the full mapped tool response
#   - created_at lets us age out rows in the future without a migration
#   - new columns can be added with ALTER TABLE without breaking reads
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS wiki_cache (
    endpoint      TEXT NOT NULL,
    query         TEXT NOT NULL,
    response_json TEXT NOT NULL,
    created_at    REAL NOT NULL,
    PRIMARY KEY (endpoint, query)
);
"""


def set_cache_path(path: str | os.PathLike[str]) -> None:
    """Point the module at a different SQLite file.

    Mainly for tests (so we never write to the real cache from pytest).
    Resets the init flag so the new file is created on next access.
    """
    global _cache_path, _cache_initialized_for
    _cache_path = Path(path)
    _cache_initialized_for = None


def _ensure_cache() -> sqlite3.Connection:
    """Open the cache, creating the directory and schema if needed."""
    global _cache_initialized_for
    with _cache_init_lock:
        if _cache_initialized_for != _cache_path:
            _cache_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(_cache_path))
            conn.execute(_SCHEMA_SQL)
            conn.commit()
            conn.close()
            _cache_initialized_for = _cache_path
    # Per-call connection — sqlite3 connections are not thread-safe to share
    # by default, and search_wikipedia may be called from worker threads.
    return sqlite3.connect(str(_cache_path))


def _cache_get(endpoint: str, query: str) -> dict[str, Any] | None:
    conn = _ensure_cache()
    try:
        row = conn.execute(
            "SELECT response_json FROM wiki_cache WHERE endpoint = ? AND query = ?",
            (endpoint, query),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    try:
        return json.loads(row[0])
    except json.JSONDecodeError:
        # Corrupt row — treat as cache miss; next call will overwrite.
        return None


def _cache_put(endpoint: str, query: str, response: dict[str, Any]) -> None:
    conn = _ensure_cache()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO wiki_cache "
            "(endpoint, query, response_json, created_at) VALUES (?, ?, ?, ?)",
            (endpoint, query, json.dumps(response), time.time()),
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# HTTP with exponential backoff + jitter
# ---------------------------------------------------------------------------

# Sentinel exception class used internally to signal "all retries exhausted".
class _WikiTransportError(Exception):
    pass


def _sleep_backoff(attempt: int) -> None:
    """Sleep ``base * 2**attempt`` plus 0..base jitter."""
    delay = _BASE_BACKOFF_S * (2**attempt) + random.uniform(0, _BASE_BACKOFF_S)
    time.sleep(delay)


def _http_get_json(params: dict[str, Any]) -> dict[str, Any]:
    """GET ``WIKIPEDIA_ENDPOINT`` with retries on 429/5xx. Raises on hard fail."""
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    last_status: int | None = None
    last_err: str | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.get(
                WIKIPEDIA_ENDPOINT,
                params=params,
                headers=headers,
                timeout=_REQUEST_TIMEOUT_S,
            )
        except requests.RequestException as e:  # network-level failure
            last_err = f"network: {type(e).__name__}: {e}"
            if attempt < _MAX_RETRIES - 1:
                _sleep_backoff(attempt)
                continue
            raise _WikiTransportError(last_err) from e

        status = resp.status_code
        last_status = status
        if status == 200:
            try:
                return resp.json()
            except ValueError as e:
                raise _WikiTransportError(f"invalid JSON from MediaWiki: {e}") from e
        if status == 429 or 500 <= status < 600:
            if attempt < _MAX_RETRIES - 1:
                _sleep_backoff(attempt)
                continue
            raise _WikiTransportError(f"http {status} after {_MAX_RETRIES} retries")
        # Other 4xx — non-retryable.
        raise _WikiTransportError(f"http {status}")
    # Unreachable, but keep the type checker happy.
    raise _WikiTransportError(
        f"exhausted retries (last_status={last_status}, last_err={last_err})"
    )


# ---------------------------------------------------------------------------
# Public tool
# ---------------------------------------------------------------------------

# Locked endpoint params (PLANS.md "Decisions" section).
_BASE_PARAMS: dict[str, Any] = {
    "action": "query",
    "generator": "search",
    "gsrlimit": 1,
    "prop": "extracts|info",
    "exintro": 1,
    "explaintext": 1,
    "inprop": "url",
    "format": "json",
    "formatversion": 2,
}


def _no_hit(error: str) -> dict[str, Any]:
    return {"title": None, "url": None, "lead": "", "error": error}


def _map_response(payload: dict[str, Any]) -> dict[str, Any]:
    """Map MediaWiki Action API response to the tool's return shape."""
    query = payload.get("query") or {}
    pages = query.get("pages") or []
    if not pages:
        return _no_hit("no_results")
    page = pages[0]
    # MediaWiki sometimes returns "missing": True for stub entries.
    if page.get("missing"):
        return _no_hit("no_results")
    title = page.get("title")
    url = page.get("fullurl")
    lead = page.get("extract") or ""
    if not title or not url:
        return _no_hit("malformed_response")
    return {"title": title, "url": url, "lead": lead, "error": None}


def search_wikipedia(query: str) -> dict[str, Any]:
    """Search Wikipedia for ``query`` and return the lead of the top hit.

    Locked endpoint pattern (one round trip): generator=search → prop=extracts.

    Cached on (endpoint, query). Never raises — failures surface in the
    ``error`` field of the returned dict.
    """
    if not isinstance(query, str) or not query.strip():
        return _no_hit("empty_query")

    cached = _cache_get(WIKIPEDIA_ENDPOINT, query)
    if cached is not None:
        return cached

    params = dict(_BASE_PARAMS)
    params["gsrsearch"] = query

    try:
        payload = _http_get_json(params)
    except _WikiTransportError as e:
        # Don't cache transport failures — caller may retry later and we'd
        # rather hit the network again than serve a stale error.
        return _no_hit(str(e))
    except Exception as e:  # noqa: BLE001 — contract: never raise
        return _no_hit(f"unexpected: {type(e).__name__}: {e}")

    response = _map_response(payload)
    # Cache the mapped response — even no-hits, so we don't hammer the API
    # for the same dead query. Transport errors above are *not* cached.
    _cache_put(WIKIPEDIA_ENDPOINT, query, response)
    return response


# ---------------------------------------------------------------------------
# Smoke / debug helper
# ---------------------------------------------------------------------------

def _smoke(query: str = "Severance (TV series)") -> None:  # pragma: no cover
    """Run a real query — for human inspection only."""
    result = search_wikipedia(query)
    print(json.dumps(result, indent=2)[:1000])


if __name__ == "__main__":  # pragma: no cover
    import sys

    _smoke(sys.argv[1] if len(sys.argv) > 1 else "Severance (TV series)")


# ---------------------------------------------------------------------------
# URL helper exposed for tests (build the URL we *would* request)
# ---------------------------------------------------------------------------

def _debug_build_url(query: str) -> str:
    """Return the fully-qualified URL we'd hit for ``query`` (debug only)."""
    params = dict(_BASE_PARAMS)
    params["gsrsearch"] = query
    return f"{WIKIPEDIA_ENDPOINT}?{urlencode(params)}"
