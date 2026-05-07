"""
tests/test_gen_stable.py

Unit tests for datagen/gen_stable.py.
All Anthropic API calls are mocked — no real network calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from datagen.gen_stable import (
    _jaccard,
    _ngrams,
    _parse_records,
    _validate_record,
    dedup_records,
    generate,
)


# ---------------------------------------------------------------------------
# Unit tests — pure functions
# ---------------------------------------------------------------------------


class TestNgrams:
    def test_basic(self):
        grams = _ngrams("hello world", 5)
        assert isinstance(grams, set)
        assert len(grams) > 0

    def test_short_string(self):
        grams = _ngrams("abc", 5)
        assert grams == {"abc"}

    def test_normalises_whitespace(self):
        grams1 = _ngrams("hello  world", 5)
        grams2 = _ngrams("hello world", 5)
        assert grams1 == grams2


class TestJaccard:
    def test_same_string(self):
        assert _jaccard("same", "same") == pytest.approx(1.0)

    def test_empty_strings(self):
        # both empty → both become empty sets → handled
        result = _jaccard("", "")
        assert 0.0 <= result <= 1.0

    def test_asymmetric(self):
        a = "Who wrote Hamlet?"
        b = "Who wrote Don Quixote?"
        j = _jaccard(a, b)
        assert 0 < j < 1


class TestDedupRecords:
    def test_removes_exact_duplicate(self):
        records = [
            {"user_question": "Who wrote Hamlet?"},
            {"user_question": "Who wrote Hamlet?"},
        ]
        result = dedup_records(records)
        assert len(result) == 1

    def test_keeps_different(self):
        records = [
            {"user_question": "Who wrote Hamlet?"},
            {"user_question": "What is the speed of light?"},
        ]
        result = dedup_records(records)
        assert len(result) == 2

    def test_empty_input(self):
        assert dedup_records([]) == []

    def test_single_item(self):
        records = [{"user_question": "Is the earth round?"}]
        result = dedup_records(records)
        assert len(result) == 1


class TestParseRecords:
    def test_plain_json_array(self):
        data = [
            {"user_question": "Who wrote Hamlet?", "golden_answer": "Shakespeare wrote Hamlet."},
            {"user_question": "What is H2O?", "golden_answer": "Water."},
        ]
        text = json.dumps(data)
        result = _parse_records(text)
        assert len(result) == 2
        assert result[0]["user_question"] == "Who wrote Hamlet?"

    def test_markdown_fenced_json(self):
        data = [{"user_question": "Q?", "golden_answer": "A."}]
        text = f"```json\n{json.dumps(data)}\n```"
        result = _parse_records(text)
        assert len(result) == 1

    def test_json_array_embedded_in_text(self):
        data = [{"user_question": "Q?", "golden_answer": "A."}]
        text = f"Here is the output:\n{json.dumps(data)}\nEnd."
        result = _parse_records(text)
        assert len(result) == 1

    def test_invalid_returns_empty(self):
        result = _parse_records("This is not JSON at all.")
        assert result == []

    def test_empty_array(self):
        result = _parse_records("[]")
        assert result == []


class TestValidateRecord:
    def test_valid_record(self):
        rec = {"user_question": "Who wrote Hamlet?", "golden_answer": "William Shakespeare."}
        assert _validate_record(rec) is True

    def test_missing_question(self):
        rec = {"golden_answer": "Some answer."}
        assert _validate_record(rec) is False

    def test_missing_answer(self):
        rec = {"user_question": "Who?"}
        assert _validate_record(rec) is False

    def test_question_without_question_mark(self):
        rec = {"user_question": "Who wrote Hamlet", "golden_answer": "Shakespeare."}
        assert _validate_record(rec) is False

    def test_not_a_dict(self):
        assert _validate_record("not a dict") is False  # type: ignore[arg-type]
        assert _validate_record(None) is False  # type: ignore[arg-type]
        assert _validate_record(42) is False  # type: ignore[arg-type]

    def test_empty_question(self):
        rec = {"user_question": "", "golden_answer": "Answer."}
        assert _validate_record(rec) is False


# ---------------------------------------------------------------------------
# Integration test — generate() with mocked Anthropic client
# ---------------------------------------------------------------------------


def _make_mock_client(records: list[dict]) -> MagicMock:
    """Build a mock Anthropic client that returns `records` as a JSON array."""
    client = MagicMock()
    resp = MagicMock()
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = json.dumps(records)
    resp.content = [text_block]
    client.messages.create.return_value = resp
    return client


class TestGenerate:
    def test_basic_generate(self, tmp_path):
        """generate() writes correct JSONL and returns expected number of records."""
        out_file = tmp_path / "tool_not_required.jsonl"
        target = 3

        # Use deliberately diverse phrasings — the production dedup catches
        # near-duplicates by n-gram overlap, so uniform fixtures would collapse.
        diverse_questions = [
            ("Who wrote Pride and Prejudice?", "Jane Austen."),
            ("What is the capital of Mongolia?", "Ulaanbaatar."),
            ("How many planets are in our solar system?", "Eight."),
            ("Which element has atomic number 79?", "Gold."),
            ("Who painted Starry Night?", "Vincent van Gogh."),
            ("What year did the Titanic sink?", "1912."),
            ("Who composed the Brandenburg Concertos?", "Johann Sebastian Bach."),
            ("What is the boiling point of water in Celsius?", "100 degrees."),
            ("Which mountain is the tallest on Earth?", "Mount Everest."),
            ("Who invented the printing press?", "Johannes Gutenberg."),
        ]
        mock_records = [
            {"user_question": q, "golden_answer": a} for q, a in diverse_questions
        ]
        client = _make_mock_client(mock_records)

        result = generate(str(out_file), target=target, client=client)

        assert len(result) == target
        assert out_file.exists()

        lines = [json.loads(l) for l in out_file.read_text().strip().split("\n") if l.strip()]
        assert len(lines) == target
        for i, rec in enumerate(lines):
            assert rec["id"] == f"tnr_{i:04d}"
            assert "user_question" in rec
            assert "golden_answer" in rec

    def test_dedup_applied(self, tmp_path):
        """generate() removes near-duplicate questions."""
        out_file = tmp_path / "out.jsonl"
        identical_q = "Who discovered gravity?"
        mock_records = [
            {"user_question": identical_q, "golden_answer": "Newton."}
            for _ in range(10)
        ]
        client = _make_mock_client(mock_records)

        result = generate(str(out_file), target=5, client=client)

        # After dedup only 1 unique question remains
        assert len(result) == 1

    def test_invalid_records_filtered(self, tmp_path):
        """generate() filters out records missing required fields."""
        out_file = tmp_path / "out.jsonl"
        mock_records = [
            {"user_question": "Good question?", "golden_answer": "Good answer."},  # valid
            {"user_question": "Missing answer"},                                    # invalid — no answer
            {"golden_answer": "Missing question."},                                 # invalid — no question
            {"user_question": "No question mark", "golden_answer": "Ans."},        # invalid — no ?
            {"user_question": "Another good one?", "golden_answer": "Fine."},      # valid
        ]
        client = _make_mock_client(mock_records)

        result = generate(str(out_file), target=5, client=client)

        # Only 2 valid records
        assert len(result) == 2

    def test_ids_assigned(self, tmp_path):
        """generate() assigns sequential tnr_XXXX IDs."""
        out_file = tmp_path / "out.jsonl"
        # Diverse questions — uniform "Question {i}?" trips the dedup filter.
        question_pairs = [
            ("Who wrote Hamlet?", "Shakespeare."),
            ("What is the capital of Peru?", "Lima."),
            ("Which planet is closest to the sun?", "Mercury."),
            ("Who painted the Mona Lisa?", "Leonardo da Vinci."),
            ("What is the speed of light?", "About 299,792 km/s."),
        ]
        mock_records = [
            {"user_question": q, "golden_answer": a} for q, a in question_pairs
        ]
        client = _make_mock_client(mock_records)
        result = generate(str(out_file), target=5, client=client)
        ids = [r["id"] for r in result]
        assert ids == ["tnr_0000", "tnr_0001", "tnr_0002", "tnr_0003", "tnr_0004"]

    def test_output_file_created(self, tmp_path):
        """generate() creates the output directory and file if needed."""
        nested_out = tmp_path / "nested" / "dir" / "output.jsonl"
        mock_records = [{"user_question": "Q?", "golden_answer": "A."}]
        client = _make_mock_client(mock_records)
        generate(str(nested_out), target=1, client=client)
        assert nested_out.exists()

    def test_model_called_with_correct_params(self, tmp_path):
        """generate() calls the Anthropic client with the correct model and no tools."""
        out_file = tmp_path / "out.jsonl"
        mock_records = [{"user_question": "Q?", "golden_answer": "A."}]
        client = _make_mock_client(mock_records)

        generate(str(out_file), target=1, client=client)

        client.messages.create.assert_called_once()
        call_kwargs = client.messages.create.call_args
        # Should not include tools= keyword
        assert "tools" not in (call_kwargs.kwargs or {})
        # Should call the right model
        assert call_kwargs.kwargs.get("model") == "claude-opus-4-7" or \
               (call_kwargs.args and call_kwargs.args[0] == "claude-opus-4-7")

    def test_empty_response_returns_empty(self, tmp_path):
        """generate() handles empty or invalid response gracefully."""
        out_file = tmp_path / "out.jsonl"
        client = MagicMock()
        resp = MagicMock()
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Not valid JSON at all."
        resp.content = [text_block]
        client.messages.create.return_value = resp

        result = generate(str(out_file), target=5, client=client)
        assert result == []
        assert out_file.exists()
        assert out_file.read_text().strip() == ""
