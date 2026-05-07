"""Unit tests for ``agent.haiku_agent.run_agent``.

Anthropic SDK and ``search_wikipedia`` are both mocked — no network, no API
key required. We script a sequence of fake assistant responses (text / tool_use)
and assert the loop walks them correctly, hits the 5-call cap, and emits a
trace conforming to the PLANS.md contract.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from agent import haiku_agent
from agent.haiku_agent import MAX_TOOL_CALLS, TOOL_NAME, run_agent
from agent.prompts import SYSTEM_PROMPTS, TOOL_DESCRIPTIONS


# ---------------------------------------------------------------------------
# Helpers — fake Anthropic response shapes
# ---------------------------------------------------------------------------


class _Block:
    """Minimal stand-in for SDK content blocks (TextBlock / ToolUseBlock)."""

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


def _text_block(text: str) -> _Block:
    return _Block(type="text", text=text)


def _tool_use_block(tool_id: str, query: str, name: str = TOOL_NAME) -> _Block:
    return _Block(type="tool_use", id=tool_id, name=name, input={"query": query})


def _resp(blocks: list[_Block], stop_reason: str = "end_turn") -> MagicMock:
    m = MagicMock()
    m.content = blocks
    m.stop_reason = stop_reason
    return m


def _make_client(responses: list[MagicMock]) -> MagicMock:
    client = MagicMock()
    client.messages.create = MagicMock(side_effect=responses)
    return client


# ---------------------------------------------------------------------------
# No tool use
# ---------------------------------------------------------------------------


def test_no_tool_use_returns_text_directly() -> None:
    client = _make_client([_resp([_text_block("Mark Twain wrote it.")])])
    tool_fn = MagicMock()

    trace = run_agent(
        "Who wrote Huck Finn?",
        tool_description_id="baseline",
        client=client,
        tool_fn=tool_fn,
    )

    assert trace["final_answer"] == "Mark Twain wrote it."
    assert trace["n_tool_calls"] == 0
    assert trace["tool_description_id"] == "baseline"
    assert trace["system_prompt_id"] == "baseline"
    assert trace["model"] == "claude-haiku-4-5"
    assert trace["latency_ms"] >= 0
    tool_fn.assert_not_called()


# ---------------------------------------------------------------------------
# Single-call happy path
# ---------------------------------------------------------------------------


def test_single_tool_call_then_final_answer() -> None:
    responses = [
        _resp(
            [_tool_use_block("tu_1", "Severance season 2 finale")],
            stop_reason="tool_use",
        ),
        _resp([_text_block("It aired on March 21, 2025.")]),
    ]
    client = _make_client(responses)
    tool_fn = MagicMock(
        return_value={
            "title": "Severance (TV series)",
            "url": "https://en.wikipedia.org/wiki/Severance_(TV_series)",
            "lead": "...the season 2 finale aired on March 21, 2025...",
            "error": None,
        }
    )

    trace = run_agent(
        "When did Severance s2 wrap up?",
        tool_description_id="baseline",
        client=client,
        tool_fn=tool_fn,
    )

    assert trace["n_tool_calls"] == 1
    assert trace["final_answer"] == "It aired on March 21, 2025."
    tool_fn.assert_called_once_with("Severance season 2 finale")

    # Trace must contain a tool_use AND a tool_result block.
    flat = [
        b
        for msg in trace["messages"]
        for b in (msg["content"] if isinstance(msg["content"], list) else [])
    ]
    assert any(b.get("type") == "tool_use" for b in flat)
    assert any(b.get("type") == "tool_result" for b in flat)


# ---------------------------------------------------------------------------
# Tool spec wiring
# ---------------------------------------------------------------------------


def test_tool_spec_uses_locked_input_schema() -> None:
    client = _make_client([_resp([_text_block("ok")])])
    run_agent("hi", tool_description_id="baseline", client=client, tool_fn=MagicMock())
    kwargs = client.messages.create.call_args.kwargs
    [tool] = kwargs["tools"]
    assert tool["name"] == TOOL_NAME
    assert tool["description"] == TOOL_DESCRIPTIONS["baseline"]
    assert tool["input_schema"] == {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }


def test_baseline_system_prompt_is_omitted_when_empty() -> None:
    """Empty-string baseline must not be sent as a system arg (SDK quirk-safe)."""
    assert SYSTEM_PROMPTS["baseline"] == ""
    client = _make_client([_resp([_text_block("ok")])])
    run_agent("hi", tool_description_id="baseline", client=client, tool_fn=MagicMock())
    kwargs = client.messages.create.call_args.kwargs
    assert "system" not in kwargs


def test_unknown_ids_raise() -> None:
    with pytest.raises(KeyError):
        run_agent(
            "x",
            tool_description_id="nope",
            client=_make_client([_resp([_text_block("")])]),
            tool_fn=MagicMock(),
        )
    with pytest.raises(KeyError):
        run_agent(
            "x",
            tool_description_id="baseline",
            system_prompt_id="nope",
            client=_make_client([_resp([_text_block("")])]),
            tool_fn=MagicMock(),
        )


# ---------------------------------------------------------------------------
# Hard cap at 5 tool calls
# ---------------------------------------------------------------------------


def test_hard_cap_strips_tools_and_forces_final_answer() -> None:
    """5 tool_use rounds → 6th call has tools removed and yields final text."""
    responses = []
    for i in range(MAX_TOOL_CALLS):
        responses.append(
            _resp([_tool_use_block(f"tu_{i}", f"q{i}")], stop_reason="tool_use")
        )
    # Final forced call (no tools) must produce text.
    responses.append(_resp([_text_block("Best guess: ...")]))

    client = _make_client(responses)
    tool_fn = MagicMock(
        return_value={"title": "T", "url": "U", "lead": "L", "error": None}
    )

    trace = run_agent(
        "edge-case",
        tool_description_id="baseline",
        client=client,
        tool_fn=tool_fn,
    )

    assert trace["n_tool_calls"] == MAX_TOOL_CALLS
    assert trace["final_answer"] == "Best guess: ..."
    # Last call must NOT include tools.
    final_kwargs = client.messages.create.call_args_list[-1].kwargs
    assert "tools" not in final_kwargs


# ---------------------------------------------------------------------------
# Multiple tool_use blocks in one assistant turn
# ---------------------------------------------------------------------------


def test_parallel_tool_uses_in_one_turn() -> None:
    responses = [
        _resp(
            [_tool_use_block("tu_a", "alpha"), _tool_use_block("tu_b", "beta")],
            stop_reason="tool_use",
        ),
        _resp([_text_block("done")]),
    ]
    client = _make_client(responses)
    tool_fn = MagicMock(
        return_value={"title": "T", "url": "U", "lead": "L", "error": None}
    )

    trace = run_agent(
        "parallel",
        tool_description_id="baseline",
        client=client,
        tool_fn=tool_fn,
    )

    assert trace["n_tool_calls"] == 2
    assert tool_fn.call_count == 2
    # The tool_result message must contain both ids in order.
    user_msgs = [m for m in trace["messages"] if m["role"] == "user"]
    # First user msg is the question (str). Second is the tool_result list.
    tool_result_msg = user_msgs[1]
    ids = [b["tool_use_id"] for b in tool_result_msg["content"]]
    assert ids == ["tu_a", "tu_b"]


# ---------------------------------------------------------------------------
# Tool failure handling
# ---------------------------------------------------------------------------


def test_tool_exception_is_captured_not_raised() -> None:
    responses = [
        _resp([_tool_use_block("tu_1", "boom")], stop_reason="tool_use"),
        _resp([_text_block("recovered")]),
    ]
    client = _make_client(responses)

    def _broken(_q: str) -> dict[str, Any]:
        raise RuntimeError("kaboom")

    trace = run_agent(
        "x",
        tool_description_id="baseline",
        client=client,
        tool_fn=_broken,
    )

    assert trace["final_answer"] == "recovered"
    # The tool_result must surface the error in its content payload.
    flat = [
        b
        for msg in trace["messages"]
        for b in (msg["content"] if isinstance(msg["content"], list) else [])
    ]
    tool_result = next(b for b in flat if b.get("type") == "tool_result")
    assert "tool_exc" in tool_result["content"]


# ---------------------------------------------------------------------------
# Ensure trace is JSON-serializable end to end
# ---------------------------------------------------------------------------


def test_trace_is_json_serializable() -> None:
    import json

    responses = [
        _resp([_tool_use_block("tu_1", "q")], stop_reason="tool_use"),
        _resp([_text_block("ok")]),
    ]
    client = _make_client(responses)
    tool_fn = MagicMock(
        return_value={"title": "T", "url": "U", "lead": "L", "error": None}
    )

    trace = run_agent(
        "serializability",
        tool_description_id="baseline",
        client=client,
        tool_fn=tool_fn,
    )
    # Must round-trip without `default=` — every block must be a plain dict.
    json.dumps(trace)
