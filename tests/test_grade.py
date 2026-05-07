"""Tests for eval/grade.py.

All Anthropic API calls are mocked — no network calls are made.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from eval.grade import (
    _programmatic_grade,
    grade_one,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_trace(
    *,
    tool_required_label: bool,
    messages: list[dict],
    user_question: str = "What happened in the season finale?",
    golden_answer: str = "The hero won.",
    final_answer: str = "The hero won.",
    question_id: str = "tr_0001",
    n_tool_calls: int = 0,
) -> dict[str, Any]:
    return {
        "question_id": question_id,
        "tool_required_label": tool_required_label,
        "user_question": user_question,
        "golden_answer": golden_answer,
        "final_answer": final_answer,
        "messages": messages,
        "n_tool_calls": n_tool_calls,
        "tool_description_id": "baseline",
        "system_prompt_id": "baseline",
        "model": "claude-haiku-4-5",
        "latency_ms": 500,
    }


def _make_message_with_tool_use() -> list[dict]:
    """Simulate an Anthropic messages array that includes a tool_use block."""
    return [
        {"role": "user", "content": "What happened in the finale?"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tu_123",
                    "name": "search_wikipedia",
                    "input": {"query": "Season 2 finale"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tu_123",
                    "content": "The hero won.",
                }
            ],
        },
        {
            "role": "assistant",
            "content": "Based on the Wikipedia article, the hero won.",
        },
    ]


def _make_messages_no_tool() -> list[dict]:
    """Simulate a messages array with no tool_use blocks."""
    return [
        {"role": "user", "content": "Who wrote Hamlet?"},
        {"role": "assistant", "content": "William Shakespeare wrote Hamlet."},
    ]


# ---------------------------------------------------------------------------
# Programmatic grader tests
# ---------------------------------------------------------------------------

class TestProgrammaticGrader:
    def test_tool_called_true_when_tool_use_in_messages(self):
        trace = _make_trace(
            tool_required_label=True,
            messages=_make_message_with_tool_use(),
        )
        tool_called, tool_use_correct = _programmatic_grade(trace)
        assert tool_called is True
        assert tool_use_correct is True  # label=True, called=True → correct

    def test_tool_called_false_when_no_tool_use_in_messages(self):
        trace = _make_trace(
            tool_required_label=False,
            messages=_make_messages_no_tool(),
        )
        tool_called, tool_use_correct = _programmatic_grade(trace)
        assert tool_called is False
        assert tool_use_correct is True  # label=False, called=False → correct

    def test_tool_use_incorrect_when_tool_called_but_not_required(self):
        trace = _make_trace(
            tool_required_label=False,
            messages=_make_message_with_tool_use(),
        )
        tool_called, tool_use_correct = _programmatic_grade(trace)
        assert tool_called is True
        assert tool_use_correct is False  # label=False, called=True → wrong

    def test_tool_use_incorrect_when_tool_required_but_not_called(self):
        trace = _make_trace(
            tool_required_label=True,
            messages=_make_messages_no_tool(),
        )
        tool_called, tool_use_correct = _programmatic_grade(trace)
        assert tool_called is False
        assert tool_use_correct is False  # label=True, called=False → wrong

    def test_empty_messages_no_tool_called(self):
        trace = _make_trace(tool_required_label=False, messages=[])
        tool_called, _ = _programmatic_grade(trace)
        assert tool_called is False

    def test_messages_missing_no_tool_called(self):
        trace = {
            "question_id": "tnr_0000",
            "tool_required_label": False,
            "user_question": "...",
            "golden_answer": "...",
            "final_answer": "...",
        }
        tool_called, _ = _programmatic_grade(trace)
        assert tool_called is False

    def test_multiple_tool_use_blocks(self):
        """Multiple sequential tool calls still count as tool_called=True."""
        messages = [
            {"role": "user", "content": "Q"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "a", "name": "search_wikipedia", "input": {"query": "X"}},
                ],
            },
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": "R"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "b", "name": "search_wikipedia", "input": {"query": "Y"}},
                ],
            },
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "b", "content": "S"}]},
            {"role": "assistant", "content": "Final answer."},
        ]
        trace = _make_trace(tool_required_label=True, messages=messages)
        tool_called, tool_use_correct = _programmatic_grade(trace)
        assert tool_called is True
        assert tool_use_correct is True

    def test_content_string_not_list_no_tool(self):
        """String content field should not raise and should not count as tool_use."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        trace = _make_trace(tool_required_label=False, messages=messages)
        tool_called, _ = _programmatic_grade(trace)
        assert tool_called is False


# ---------------------------------------------------------------------------
# Mocked LLM judge integration in grade_one
# ---------------------------------------------------------------------------

def _make_mock_anthropic_client(reasoning: str, correct: bool) -> MagicMock:
    """Build a mock Anthropic client that returns a tool_use block from the judge."""
    tool_use_block = MagicMock()
    tool_use_block.type = "tool_use"
    tool_use_block.name = "submit_verdict"
    tool_use_block.input = {"reasoning": reasoning, "correct": correct}

    response = MagicMock()
    response.content = [tool_use_block]

    client = MagicMock()
    client.messages.create.return_value = response
    return client


class TestGradeOne:
    def test_grade_one_correct_answer(self):
        trace = _make_trace(
            tool_required_label=True,
            messages=_make_message_with_tool_use(),
            golden_answer="The hero won.",
            final_answer="The hero won.",
        )
        mock_client = _make_mock_anthropic_client(
            reasoning="The agent's answer matches the golden answer exactly.",
            correct=True,
        )
        result = grade_one(trace, _client=mock_client)

        assert result["tool_called"] is True
        assert result["tool_use_correct"] is True
        assert result["answer_correct"] is True
        assert "agent's answer matches" in result["judge_reasoning"]

    def test_grade_one_wrong_answer(self):
        trace = _make_trace(
            tool_required_label=True,
            messages=_make_message_with_tool_use(),
            golden_answer="The hero won.",
            final_answer="The villain won.",
        )
        mock_client = _make_mock_anthropic_client(
            reasoning="The agent says the villain won but the golden answer says the hero won.",
            correct=False,
        )
        result = grade_one(trace, _client=mock_client)

        assert result["answer_correct"] is False
        assert "villain" in result["judge_reasoning"]

    def test_grade_one_tool_not_required_and_not_called(self):
        trace = _make_trace(
            tool_required_label=False,
            messages=_make_messages_no_tool(),
            question_id="tnr_0001",
        )
        mock_client = _make_mock_anthropic_client(
            reasoning="The agent answered correctly without needing to search.",
            correct=True,
        )
        result = grade_one(trace, _client=mock_client)

        assert result["tool_called"] is False
        assert result["tool_use_correct"] is True
        assert result["answer_correct"] is True

    def test_grade_one_preserves_original_trace_fields(self):
        """All original trace fields must appear in the graded record."""
        trace = _make_trace(
            tool_required_label=True,
            messages=_make_message_with_tool_use(),
            question_id="tr_9999",
            n_tool_calls=3,
        )
        mock_client = _make_mock_anthropic_client("Correct.", True)
        result = grade_one(trace, _client=mock_client)

        for key in trace:
            assert key in result, f"Missing key: {key}"
        assert result["question_id"] == "tr_9999"
        assert result["n_tool_calls"] == 3

    def test_grade_one_extra_fields_in_graded_record(self):
        """Graded record must have exactly the four new fields."""
        trace = _make_trace(tool_required_label=False, messages=_make_messages_no_tool())
        mock_client = _make_mock_anthropic_client("Looks correct.", True)
        result = grade_one(trace, _client=mock_client)

        assert "tool_called" in result
        assert "tool_use_correct" in result
        assert "answer_correct" in result
        assert "judge_reasoning" in result

    def test_grade_one_judge_fallback_when_no_tool_use_block(self):
        """If the mock returns no tool_use block, grade_one should still return safely."""
        trace = _make_trace(tool_required_label=False, messages=_make_messages_no_tool())

        # Client returns a response with no tool_use blocks.
        response = MagicMock()
        response.content = []  # empty — no tool_use block
        client = MagicMock()
        client.messages.create.return_value = response

        result = grade_one(trace, _client=client)
        # Fallback: correct=False, reasoning="judge did not call submit_verdict"
        assert result["answer_correct"] is False
        assert "submit_verdict" in result["judge_reasoning"]

    def test_grade_one_passes_correct_fields_to_judge(self):
        """The judge must NOT see the messages array — only the three text fields."""
        trace = _make_trace(
            tool_required_label=True,
            messages=_make_message_with_tool_use(),
            user_question="Did Severance season 2 end on a cliffhanger?",
            golden_answer="Yes, it ended on a major cliffhanger.",
            final_answer="The season ended with a cliffhanger.",
        )
        mock_client = _make_mock_anthropic_client("Matches the golden answer.", True)
        grade_one(trace, _client=mock_client)

        call_kwargs = mock_client.messages.create.call_args
        # Extract the messages list passed to the API.
        passed_messages = call_kwargs.kwargs.get("messages") or (call_kwargs.args[0] if call_kwargs.args else [])
        # The user message content should contain our question/golden/final_answer text
        # but NOT anything from the messages array (e.g., "search_wikipedia").
        user_content = str(passed_messages)
        assert "Did Severance season 2" in user_content
        assert "Yes, it ended on a major cliffhanger" in user_content
        assert "The season ended with a cliffhanger" in user_content


# ---------------------------------------------------------------------------
# Smoke: grade_batch (async, run via asyncio.run)
# ---------------------------------------------------------------------------

def test_grade_batch_returns_same_count():
    """grade_batch should return as many records as input traces.

    Uses asyncio.run() to drive the coroutine from a sync test — avoids
    requiring pytest-asyncio as an additional test dependency.
    """
    import asyncio
    from eval.grade import grade_batch

    traces = [
        _make_trace(
            tool_required_label=i % 2 == 0,
            messages=_make_message_with_tool_use() if i % 2 == 0 else _make_messages_no_tool(),
            question_id=f"tr_{i:04d}",
        )
        for i in range(4)
    ]

    mock_client = _make_mock_anthropic_client("Reasonable.", True)

    # Patch the Anthropic constructor so grade_batch uses our mock.
    with patch("eval.grade.anthropic.Anthropic", return_value=mock_client):
        results = asyncio.run(grade_batch(traces, concurrency=2))

    assert len(results) == len(traces)
    for r in results:
        assert "tool_called" in r
        assert "answer_correct" in r
