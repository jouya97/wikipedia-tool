"""Unit tests for ``wiki.client.search_wikipedia``.

NO real network. All HTTP is mocked via ``unittest.mock.patch`` on
``wiki.client.requests.get`` (stdlib mocking — avoids the requests-mock
dependency). Cache is redirected at a tmp_path so the production cache
file is never touched.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from wiki import client as wiki_client
from wiki.client import (
    USER_AGENT,
    WIKIPEDIA_ENDPOINT,
    search_wikipedia,
    set_cache_path,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the cache at tmp_path and shrink the backoff so tests are fast."""
    set_cache_path(tmp_path / "wikipedia.sqlite")
    monkeypatch.setattr(wiki_client, "_BASE_BACKOFF_S", 0.0)
    monkeypatch.setattr(wiki_client, "_MAX_RETRIES", 3)
    yield
    # Reset to default so other tests don't see the tmp file.
    set_cache_path(wiki_client._DEFAULT_CACHE_PATH)


def _ok_response(payload: dict) -> MagicMock:
    m = MagicMock()
    m.status_code = 200
    m.json.return_value = payload
    return m


def _status_response(status: int) -> MagicMock:
    m = MagicMock()
    m.status_code = status
    m.json.return_value = {}
    return m


def _hit_payload(
    title: str = "Severance (TV series)",
    url: str = "https://en.wikipedia.org/wiki/Severance_(TV_series)",
    extract: str = "Severance is an American science fiction psychological thriller television series.",
) -> dict:
    return {
        "query": {
            "pages": [
                {
                    "pageid": 12345,
                    "title": title,
                    "fullurl": url,
                    "extract": extract,
                }
            ]
        }
    }


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_returns_top_hit_mapped_correctly() -> None:
    with patch("wiki.client.requests.get", return_value=_ok_response(_hit_payload())) as g:
        result = search_wikipedia("Severance TV series")

    assert result["title"] == "Severance (TV series)"
    assert result["url"].startswith("https://en.wikipedia.org/")
    assert "psychological thriller" in result["lead"]
    assert result["error"] is None

    # Endpoint + user-agent + key params.
    args, kwargs = g.call_args
    assert args[0] == WIKIPEDIA_ENDPOINT
    assert kwargs["headers"]["User-Agent"] == USER_AGENT
    params = kwargs["params"]
    assert params["action"] == "query"
    assert params["generator"] == "search"
    assert params["gsrsearch"] == "Severance TV series"
    assert params["gsrlimit"] == 1
    assert params["prop"] == "extracts|info"
    assert params["exintro"] == 1
    assert params["explaintext"] == 1
    assert params["inprop"] == "url"
    assert params["format"] == "json"
    assert params["formatversion"] == 2


# ---------------------------------------------------------------------------
# No-hit / malformed
# ---------------------------------------------------------------------------


def test_no_results_returns_no_hit_shape() -> None:
    with patch("wiki.client.requests.get", return_value=_ok_response({"batchcomplete": True})):
        result = search_wikipedia("asdfqwertyzxcv-no-such-page-12345")
    assert result == {"title": None, "url": None, "lead": "", "error": "no_results"}


def test_empty_query_short_circuits() -> None:
    with patch("wiki.client.requests.get") as g:
        result = search_wikipedia("   ")
    assert result["error"] == "empty_query"
    g.assert_not_called()


def test_missing_page_flag_treated_as_no_hit() -> None:
    payload = {"query": {"pages": [{"missing": True, "title": "Foo"}]}}
    with patch("wiki.client.requests.get", return_value=_ok_response(payload)):
        result = search_wikipedia("ghost-page")
    assert result["error"] == "no_results"


def test_malformed_page_no_url_returns_error() -> None:
    payload = {"query": {"pages": [{"title": "Only Title"}]}}
    with patch("wiki.client.requests.get", return_value=_ok_response(payload)):
        result = search_wikipedia("weird-page")
    assert result["error"] == "malformed_response"


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def test_cache_hit_skips_network() -> None:
    with patch("wiki.client.requests.get", return_value=_ok_response(_hit_payload())) as g:
        first = search_wikipedia("Severance")
        second = search_wikipedia("Severance")

    assert first == second
    assert g.call_count == 1


def test_no_hit_results_are_cached() -> None:
    """Repeated dead queries shouldn't re-hit the API."""
    with patch(
        "wiki.client.requests.get", return_value=_ok_response({"batchcomplete": True})
    ) as g:
        search_wikipedia("dead-query-xyz")
        search_wikipedia("dead-query-xyz")
    assert g.call_count == 1


# ---------------------------------------------------------------------------
# Backoff / retries
# ---------------------------------------------------------------------------


def test_429_then_success_retries() -> None:
    seq = [_status_response(429), _status_response(503), _ok_response(_hit_payload())]
    with patch("wiki.client.requests.get", side_effect=seq) as g:
        result = search_wikipedia("Severance")
    assert result["error"] is None
    assert g.call_count == 3


def test_persistent_5xx_returns_error_does_not_raise() -> None:
    with patch(
        "wiki.client.requests.get",
        side_effect=[_status_response(500)] * 10,
    ):
        result = search_wikipedia("flaky-query")
    assert result["title"] is None
    assert result["lead"] == ""
    assert "http 500" in result["error"]


def test_non_retryable_4xx_returns_error_does_not_raise() -> None:
    with patch("wiki.client.requests.get", return_value=_status_response(403)) as g:
        result = search_wikipedia("forbidden")
    assert "http 403" in result["error"]
    # Non-retryable: should fail on the first try.
    assert g.call_count == 1


def test_network_exception_returns_error() -> None:
    import requests as _requests

    with patch(
        "wiki.client.requests.get",
        side_effect=_requests.ConnectionError("boom"),
    ):
        result = search_wikipedia("offline")
    assert result["title"] is None
    assert "network" in result["error"]


def test_transport_failures_are_not_cached() -> None:
    """A failed call must not poison the cache for later retries."""
    import requests as _requests

    seq_fail = [_requests.ConnectionError("boom")] * 10
    with patch("wiki.client.requests.get", side_effect=seq_fail):
        first = search_wikipedia("retryable")
    assert first["error"] is not None

    with patch("wiki.client.requests.get", return_value=_ok_response(_hit_payload())):
        second = search_wikipedia("retryable")
    assert second["error"] is None


# ---------------------------------------------------------------------------
# Cache schema is forward-compatible
# ---------------------------------------------------------------------------


def test_cache_schema_pk_is_endpoint_query(tmp_path: Path) -> None:
    """Adding a column later must not break existing reads."""
    import sqlite3

    set_cache_path(tmp_path / "fwd.sqlite")
    with patch("wiki.client.requests.get", return_value=_ok_response(_hit_payload())):
        search_wikipedia("Severance")

    conn = sqlite3.connect(str(tmp_path / "fwd.sqlite"))
    try:
        # Simulate a future migration adding a column.
        conn.execute("ALTER TABLE wiki_cache ADD COLUMN tags TEXT")
        conn.commit()
    finally:
        conn.close()

    # Reset init flag so the module re-opens cleanly.
    wiki_client._cache_initialized_for = None
    with patch("wiki.client.requests.get") as g:
        cached = search_wikipedia("Severance")
    assert cached["title"] == "Severance (TV series)"
    g.assert_not_called()  # served from cache, post-migration
