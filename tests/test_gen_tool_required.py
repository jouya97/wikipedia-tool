"""
tests/test_gen_tool_required.py

Unit tests for datagen/gen_tool_required.py.
All Anthropic API calls and wiki.client are mocked.
"""

from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Mock wiki.client BEFORE importing datagen so the lazy import resolves
# ---------------------------------------------------------------------------


def _make_wiki_module():
    """Return a fake wiki package + wiki.client module."""
    wiki_pkg = types.ModuleType("wiki")
    wiki_client = types.ModuleType("wiki.client")

    def search_wikipedia(query: str) -> dict:
        return {
            "title": f"Mock article for {query}",
            "url": f"https://en.wikipedia.org/wiki/Mock_{query.replace(' ', '_')}",
            "lead": (
                f"This is a mock Wikipedia lead for '{query}'. "
                "It contains post-March-2025 facts about the topic."
            ),
            "error": None,
        }

    wiki_client.search_wikipedia = search_wikipedia
    wiki_pkg.client = wiki_client
    return wiki_pkg, wiki_client


# Inject the mock wiki module per-test via an autouse fixture so it doesn't
# leak into sibling test files (notably test_wiki_client.py, which needs the
# real module). datagen's lazy import inside _call_search_wikipedia resolves
# at test-execution time, so a function-scoped fixture is sufficient.
@pytest.fixture(autouse=True)
def _mock_wiki_module(monkeypatch):
    wiki_pkg, wiki_client = _make_wiki_module()
    monkeypatch.setitem(sys.modules, "wiki", wiki_pkg)
    monkeypatch.setitem(sys.modules, "wiki.client", wiki_client)
    yield


# Safe to import datagen at module level — gen_tool_required does NOT import
# wiki.client at module load (the import is lazy inside _call_search_wikipedia).
from datagen.gen_tool_required import (  # noqa: E402
    _extract_json,
    _jaccard,
    _ngrams,
    _validate_record,
    _worker,
    dedup_records,
    generate,
    load_seeds,
    partition_seeds,
)


# ---------------------------------------------------------------------------
# Pure-function unit tests
# ---------------------------------------------------------------------------


class TestNgrams:
    def test_basic(self):
        grams = _ngrams("hello world", 5)
        assert "hello" in grams

    def test_short_string(self):
        grams = _ngrams("ab", 5)
        assert grams == {"ab"}

    def test_empty_string(self):
        grams = _ngrams("", 5)
        assert isinstance(grams, set)

    def test_normalises_whitespace(self):
        g1 = _ngrams("hello  world", 5)
        g2 = _ngrams("hello world", 5)
        assert g1 == g2


class TestJaccard:
    def test_identical(self):
        assert _jaccard("hello world", "hello world") == pytest.approx(1.0)

    def test_completely_different(self):
        j = _jaccard("aaaaaaaaa", "xxxxxxxxx")
        assert j == pytest.approx(0.0)

    def test_partial_overlap(self):
        j = _jaccard("what is the capital of France", "what is the capital of Germany")
        assert 0 < j < 1

    def test_near_duplicate_threshold(self):
        a = "Did you hear about the new season of Severance?"
        b = "Did you hear about the new season of Severance?"
        assert _jaccard(a, b) > 0.5

    def test_different_questions(self):
        a = "Who won the championship in 2024?"
        b = "What did Shakespeare write in his early career?"
        assert _jaccard(a, b) < 0.5


class TestDedupRecords:
    def test_removes_exact_duplicate(self):
        records = [
            {"user_question": "Did you hear about the new season of Severance recently?"},
            {"user_question": "Did you hear about the new season of Severance recently?"},
        ]
        result = dedup_records(records)
        assert len(result) == 1

    def test_keeps_distinct(self):
        records = [
            {"user_question": "Who won the championship last year?"},
            {"user_question": "What is the boiling point of water?"},
        ]
        result = dedup_records(records)
        assert len(result) == 2

    def test_empty(self):
        assert dedup_records([]) == []

    def test_first_occurrence_kept(self):
        records = [
            {"user_question": "Is the new Severance season out yet?", "id": "first"},
            {"user_question": "Is the new Severance season out yet?", "id": "second"},
        ]
        result = dedup_records(records)
        assert result[0]["id"] == "first"


class TestExtractJson:
    def test_plain_json(self):
        text = '{"seed_title": "Test", "user_question": "Did X happen?", "golden_answer": "Yes."}'
        result = _extract_json(text)
        assert result is not None
        assert result["seed_title"] == "Test"

    def test_json_with_markdown_fence(self):
        text = '```json\n{"key": "value"}\n```'
        result = _extract_json(text)
        assert result is not None
        assert result["key"] == "value"

    def test_json_embedded_in_text(self):
        text = 'Here is output: {"seed_title": "X", "user_question": "Y?", "golden_answer": "Z."}'
        result = _extract_json(text)
        assert result is not None

    def test_invalid_json(self):
        result = _extract_json("not json at all")
        assert result is None


class TestValidateRecord:
    def test_valid(self):
        rec = {
            "seed_title": "Test",
            "user_question": "Did something happen?",
            "golden_answer": "Yes, it did.",
        }
        assert _validate_record(rec) is True

    def test_missing_question(self):
        rec = {"seed_title": "Test", "golden_answer": "Yes."}
        assert _validate_record(rec) is False

    def test_question_no_question_mark(self):
        rec = {
            "seed_title": "Test",
            "user_question": "No question mark here",
            "golden_answer": "Yes.",
        }
        assert _validate_record(rec) is False

    def test_missing_answer(self):
        rec = {"seed_title": "Test", "user_question": "Question?"}
        assert _validate_record(rec) is False


class TestLoadSeeds:
    def test_load(self, tmp_path):
        seeds_file = tmp_path / "seeds.jsonl"
        seeds_file.write_text(
            json.dumps({"title": "Show A", "topic_slice": 0, "post_march_2025_fact": "S2 aired"})
            + "\n"
            + json.dumps({"title": "Show B", "topic_slice": 1, "post_march_2025_fact": "Won award"})
            + "\n"
        )
        seeds = load_seeds(str(seeds_file))
        assert len(seeds) == 2
        assert seeds[0]["title"] == "Show A"

    def test_ignores_blank_lines(self, tmp_path):
        seeds_file = tmp_path / "seeds.jsonl"
        seeds_file.write_text(
            json.dumps({"title": "A", "topic_slice": 0}) + "\n\n"
            + json.dumps({"title": "B", "topic_slice": 1}) + "\n"
        )
        seeds = load_seeds(str(seeds_file))
        assert len(seeds) == 2


class TestPartitionSeeds:
    def test_basic_partition(self):
        seeds = [{"title": f"S{i}", "topic_slice": i % 3} for i in range(6)]
        partitions = partition_seeds(seeds, n_workers=3)
        assert len(partitions) == 3
        for p in partitions:
            assert len(p) == 2

    def test_empty_partition(self):
        seeds = [{"title": "S0", "topic_slice": 0}]
        partitions = partition_seeds(seeds, n_workers=3)
        assert len(partitions[0]) == 1
        assert len(partitions[1]) == 0
        assert len(partitions[2]) == 0


# ---------------------------------------------------------------------------
# Helpers for building mock Anthropic responses
# ---------------------------------------------------------------------------


def _tool_block(query: str, tool_id: str = "tu_001") -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = "search_wikipedia"
    block.id = tool_id
    block.input = {"query": query}
    return block


def _text_block(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _response(blocks) -> MagicMock:
    resp = MagicMock()
    resp.content = blocks
    return resp


# ---------------------------------------------------------------------------
# Worker tests (using run_in_executor passthrough patch)
# ---------------------------------------------------------------------------


def _make_run_in_executor_passthrough():
    """
    Returns an async function that executes the callable synchronously
    (ignoring the executor argument). This lets run_in_executor work in tests
    without a real thread pool.
    """
    async def fake_run_in_executor(_executor, fn, *args):
        if args:
            return fn(*args)
        return fn()
    return fake_run_in_executor


class TestWorker:
    def test_single_seed_success(self):
        """_worker produces one success for a single seed with one tool call."""
        seed = {
            "title": "TestShow",
            "topic_slice": 0,
            "post_march_2025_fact": "Season 3 premiered May 2025.",
            "category": "tv",
        }

        answer_json = json.dumps({
            "seed_title": "TestShow",
            "user_question": "Wait, is the new season of TestShow actually out?",
            "golden_answer": "Yes, Season 3 premiered in May 2025.",
            "datagen_search_queries": ["TestShow 2025"],
            "datagen_n_tool_calls": 1,
        })

        mock_client = MagicMock()
        # First call: tool use; second call: final answer
        mock_client.messages.create.side_effect = [
            _response([_tool_block("TestShow 2025", "tu_001")]),
            _response([_text_block(answer_json)]),
        ]

        async def run():
            loop = asyncio.get_running_loop()
            with patch.object(loop, "run_in_executor", side_effect=_make_run_in_executor_passthrough()):
                return await _worker(0, [seed], mock_client)

        successes, skips = asyncio.run(run())
        assert len(successes) == 1
        assert len(skips) == 0
        assert successes[0]["seed_title"] == "TestShow"
        assert successes[0]["user_question"].endswith("?")

    def test_skip_record(self):
        """_worker records a skip when model emits skip_reason."""
        seed = {"title": "UnknownTopic", "topic_slice": 0, "post_march_2025_fact": "unknown"}

        skip_json = json.dumps({"seed_title": "UnknownTopic", "skip_reason": "No Wikipedia article found."})

        mock_client = MagicMock()
        mock_client.messages.create.return_value = _response([_text_block(skip_json)])

        async def run():
            loop = asyncio.get_running_loop()
            with patch.object(loop, "run_in_executor", side_effect=_make_run_in_executor_passthrough()):
                return await _worker(0, [seed], mock_client)

        successes, skips = asyncio.run(run())
        assert len(successes) == 0
        assert len(skips) == 1
        assert "skip_reason" in skips[0]

    def test_multiple_seeds_sequential(self):
        """_worker processes multiple seeds sequentially."""
        seeds = [
            {"title": f"Show{i}", "topic_slice": 0, "post_march_2025_fact": f"fact {i}"}
            for i in range(3)
        ]

        responses = []
        for seed in seeds:
            answer_json = json.dumps({
                "seed_title": seed["title"],
                "user_question": f"What happened with {seed['title']} recently?",
                "golden_answer": f"{seed['title']} had a notable event.",
            })
            responses.append(_response([_text_block(answer_json)]))

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = responses

        async def run():
            loop = asyncio.get_running_loop()
            with patch.object(loop, "run_in_executor", side_effect=_make_run_in_executor_passthrough()):
                return await _worker(0, seeds, mock_client)

        successes, skips = asyncio.run(run())
        assert len(successes) == 3


# ---------------------------------------------------------------------------
# Integration test — generate() end-to-end
# ---------------------------------------------------------------------------


class TestGenerate:
    def _build_mock_client(self, seeds: list[dict]) -> MagicMock:
        """Each seed: one direct text response with a valid JSON answer.
        Phrasings are deliberately diverse — uniform templates would be collapsed
        by the production near-duplicate filter."""
        templates = [
            "Did you catch what happened with {t} last month?",
            "Yo is it true {t} had a wild moment recently?",
            "I keep seeing buzz about {t} — what's actually going on?",
            "Wait, didn't they shake things up on {t}?",
            "How does the latest {t} news compare to before?",
            "Friend told me {t} had a huge update — verify?",
            "Quick question, what's the deal with {t} these days?",
            "Has {t} done anything noteworthy this past season?",
        ]
        answers = [
            "{t} announced a major update after March 2025.",
            "{t} broke a notable record post March 2025.",
            "{t} had a significant cast or roster change after March 2025.",
            "{t} concluded its latest season with a milestone after March 2025.",
            "{t} was renewed or expanded following events after March 2025.",
        ]
        responses = []
        for i, seed in enumerate(seeds):
            answer_json = json.dumps({
                "seed_title": seed["title"],
                "user_question": templates[i % len(templates)].format(t=seed["title"]),
                "golden_answer": answers[i % len(answers)].format(t=seed["title"]),
            })
            responses.append(_response([_text_block(answer_json)]))
        # Extra fallback responses in case of retry
        for _ in range(10):
            responses.append(_response([_text_block(
                json.dumps({"seed_title": "extra", "skip_reason": "no more seeds"})
            )]))
        client = MagicMock()
        client.messages.create.side_effect = responses
        return client

    def test_generate_writes_jsonl(self, tmp_path):
        seeds_file = tmp_path / "seeds.jsonl"
        out_file = tmp_path / "tool_required.jsonl"
        target = 3

        seed_data = [
            {"title": f"TestShow{i}", "topic_slice": i, "post_march_2025_fact": f"fact {i}", "category": "tv"}
            for i in range(target)
        ]
        seeds_file.write_text("\n".join(json.dumps(s) for s in seed_data) + "\n")

        mock_client = self._build_mock_client(seed_data)

        async def fake_run_in_executor(_executor, fn, *args):
            return fn(*args) if args else fn()

        with patch("datagen.gen_tool_required.NUM_WORKERS", target):
            async def run():
                loop = asyncio.get_running_loop()
                with patch.object(loop, "run_in_executor", side_effect=fake_run_in_executor):
                    return await generate(
                        seeds_path=str(seeds_file),
                        out_path=str(out_file),
                        target=target,
                        client=mock_client,
                    )
            records = asyncio.run(run())

        assert out_file.exists()
        lines = [json.loads(l) for l in out_file.read_text().strip().split("\n") if l.strip()]
        assert len(lines) == len(records)
        assert len(records) == target
        assert all("id" in r for r in records)
        assert all(r["id"] == f"tr_{i:04d}" for i, r in enumerate(records))

    def test_generate_dedup_applied(self, tmp_path):
        """Identical questions get deduped, yielding fewer than target records."""
        seeds_file = tmp_path / "seeds.jsonl"
        out_file = tmp_path / "out.jsonl"

        identical_q = "Did anything happen with Show0 recently?"
        seed_data = [
            {"title": f"TestShow{i}", "topic_slice": i, "post_march_2025_fact": "fact", "category": "tv"}
            for i in range(4)
        ]
        seeds_file.write_text("\n".join(json.dumps(s) for s in seed_data) + "\n")

        # All responses produce the same user_question → dedup keeps 1
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _response([_text_block(json.dumps({
            "seed_title": "Any",
            "user_question": identical_q,
            "golden_answer": "Something happened.",
        }))])

        async def fake_run_in_executor(_executor, fn, *args):
            return fn(*args) if args else fn()

        with patch("datagen.gen_tool_required.NUM_WORKERS", 4):
            async def run():
                loop = asyncio.get_running_loop()
                with patch.object(loop, "run_in_executor", side_effect=fake_run_in_executor):
                    return await generate(
                        seeds_path=str(seeds_file),
                        out_path=str(out_file),
                        target=4,
                        client=mock_client,
                    )
            records = asyncio.run(run())

        # After dedup: only 1 unique question
        assert len(records) <= 1


class TestSkipRecord:
    """Test that skip record JSON is handled correctly by the orchestrator."""

    def test_skip_json_is_recognized(self):
        parsed = _extract_json('{"seed_title": "UnknownShow", "skip_reason": "No article found."}')
        assert parsed is not None
        assert "skip_reason" in parsed
        assert _validate_record(parsed) is False
