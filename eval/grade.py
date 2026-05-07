"""Grader for inference traces.

Two graders run in sequence for each inference trace:

1. **Programmatic grader** — deterministic, no API call.
   ``tool_called = any block of type "tool_use" in trace["messages"]``
   ``tool_use_correct = (tool_called == trace["tool_required_label"])``

2. **LLM judge** — Sonnet 4.6, low effort.
   Receives only: ``user_question``, ``golden_answer``, ``final_answer``.
   Uses Anthropic tool-use-style structured output: tool named ``submit_verdict``
   whose input_schema has ``reasoning`` FIRST (1-3 sentences) then ``correct``
   (boolean). Order matters — the judge writes its reasoning before its verdict,
   which produces more reliable correctness decisions.

Public API:
    grade_one(trace: dict) -> dict       # sync; calls Anthropic synchronously
    async grade_batch(traces, concurrency=8) -> list[dict]  # bounded async
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import anthropic
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_JUDGE_MODEL = "claude-sonnet-4-6"

# Tool definition for structured output.
# CRITICAL: "reasoning" MUST appear before "correct" in properties.
# JSON dict ordering is preserved in Python 3.7+ and the Anthropic API
# serialises properties in insertion order, so reasoning comes first.
_SUBMIT_VERDICT_TOOL: dict[str, Any] = {
    "name": "submit_verdict",
    "description": (
        "Submit a verdict on whether the agent's answer is correct. "
        "You MUST populate 'reasoning' first (1-3 sentences), then 'correct'."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": (
                    "1-3 sentences explaining your verdict. "
                    "Write this BEFORE deciding the boolean value of 'correct'."
                ),
            },
            "correct": {
                "type": "boolean",
                "description": (
                    "True if the agent's answer is substantively correct "
                    "relative to the golden answer; False otherwise."
                ),
            },
        },
        "required": ["reasoning", "correct"],
    },
}

_JUDGE_SYSTEM_PROMPT_TEMPLATE = (
    "CURRENT DATE: {today}\n"
    "Use this date to interpret relative time references in the user question "
    "or golden answer (e.g. 'this year', 'last month', 'recently').\n\n"
    "You are a strict but fair answer-quality judge. "
    "You will be given a user question, a golden (ground-truth) answer, "
    "and an agent's answer. Your job is to decide whether the agent's answer "
    "is substantively correct relative to the golden answer.\n\n"
    "Instructions:\n"
    "1. Think carefully and write 1-3 sentences of REASONING that explain your "
    "   assessment. You must write the reasoning FIRST, before deciding the verdict.\n"
    "2. Then call the `submit_verdict` tool with `reasoning` set to those sentences "
    "   and `correct` set to true or false.\n\n"
    "Grading rubric:\n"
    "- Minor wording differences, extra context, or different phrasing of the same "
    "  fact are acceptable — mark as correct.\n"
    "- Factual errors, omissions of key facts, or answers that contradict the golden "
    "  answer are incorrect.\n"
    "- If the agent says it doesn't know but the golden answer is substantive, mark "
    "  as incorrect.\n"
    "- Do NOT reward hedging or vagueness when a clear answer exists.\n"
    "- IMPORTANT — overruling stale golden answers: the golden answers were authored "
    "  in 2025 and may reference 'this year' or 'the current year' as 2025. If the "
    "  CURRENT DATE above is in a later year and the agent's answer reflects fresher "
    "  information (e.g. a more recent occurrence of the event the user asked about), "
    "  mark as correct — PROVIDED the agent's answer appears grounded in retrieved "
    "  tool results, not invented from pretraining. Sonnet is the strongest model in "
    "  this pipeline, so use your best judgment when the golden answer is the stale "
    "  one."
)


def _judge_system_prompt() -> str:
    """Build the judge system prompt with today's date inlined."""
    from datetime import datetime, timezone  # noqa: PLC0415
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return _JUDGE_SYSTEM_PROMPT_TEMPLATE.format(today=today)


def _build_judge_user_message(
    user_question: str,
    golden_answer: str,
    final_answer: str,
) -> str:
    """Build the user-turn text sent to the LLM judge."""
    return (
        f"**User question:**\n{user_question}\n\n"
        f"**Golden answer:**\n{golden_answer}\n\n"
        f"**Agent's answer:**\n{final_answer}\n\n"
        "Please write your REASONING first (1-3 sentences), then call "
        "`submit_verdict` with your reasoning and your boolean verdict."
    )


# ---------------------------------------------------------------------------
# Programmatic grader (no API call)
# ---------------------------------------------------------------------------

def _programmatic_grade(trace: dict) -> tuple[bool, bool]:
    """Return (tool_called, tool_use_correct).

    ``tool_called`` is True iff any message block has ``type == "tool_use"``.
    ``tool_use_correct`` is True iff ``tool_called == trace["tool_required_label"]``.
    """
    messages: list[dict] = trace.get("messages") or []
    tool_called = False
    for msg in messages:
        # Messages can be dicts with a "content" field that is a list of blocks,
        # or may directly have "type" set (older formats). Handle both.
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        tool_called = True
                        break
            elif isinstance(content, str):
                # Plain-text message — no tool_use block here.
                pass
            # Some trace formats store role=assistant with a list directly at
            # the message level (non-standard). Also handle the case where the
            # message itself is a tool_use block (unlikely but defensive).
            if msg.get("type") == "tool_use":
                tool_called = True
        if tool_called:
            break

    tool_required_label: bool = bool(trace.get("tool_required_label", False))
    tool_use_correct: bool = tool_called == tool_required_label
    return tool_called, tool_use_correct


# ---------------------------------------------------------------------------
# LLM judge (sync)
# ---------------------------------------------------------------------------

def _llm_judge_sync(
    user_question: str,
    golden_answer: str,
    final_answer: str,
    client: anthropic.Anthropic | None = None,
) -> tuple[bool, str]:
    """Call Sonnet 4.6 to judge answer correctness.

    Returns ``(correct: bool, reasoning: str)``.
    """
    if client is None:
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    user_message = _build_judge_user_message(user_question, golden_answer, final_answer)

    response = client.messages.create(
        model=_JUDGE_MODEL,
        max_tokens=512,
        system=_judge_system_prompt(),
        tools=[_SUBMIT_VERDICT_TOOL],
        tool_choice={"type": "any"},  # force the model to call submit_verdict
        messages=[{"role": "user", "content": user_message}],
        output_config={"effort": "low"},
    )

    # Extract the tool_use block from the response.
    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_verdict":
            inp: dict[str, Any] = block.input  # type: ignore[attr-defined]
            reasoning: str = inp.get("reasoning", "")
            correct: bool = bool(inp.get("correct", False))
            return correct, reasoning

    # Fallback — should not happen if tool_choice=any is respected.
    return False, "judge did not call submit_verdict"


# ---------------------------------------------------------------------------
# Public: grade_one (sync)
# ---------------------------------------------------------------------------

def grade_one(
    trace: dict,
    _client: anthropic.Anthropic | None = None,
) -> dict:
    """Grade a single inference trace.

    Parameters
    ----------
    trace:
        An inference trace dict conforming to the PLANS.md schema.
    _client:
        Optional pre-built ``anthropic.Anthropic`` client (used in tests to
        inject a mock; not part of the public contract).

    Returns
    -------
    dict
        The original trace fields plus:
        - ``tool_called`` (bool)
        - ``tool_use_correct`` (bool)
        - ``answer_correct`` (bool)
        - ``judge_reasoning`` (str)
    """
    tool_called, tool_use_correct = _programmatic_grade(trace)

    answer_correct, judge_reasoning = _llm_judge_sync(
        user_question=trace.get("user_question", ""),
        golden_answer=trace.get("golden_answer", ""),
        final_answer=trace.get("final_answer", ""),
        client=_client,
    )

    return {
        **trace,
        "tool_called": tool_called,
        "tool_use_correct": tool_use_correct,
        "answer_correct": answer_correct,
        "judge_reasoning": judge_reasoning,
    }


# ---------------------------------------------------------------------------
# Public: grade_batch (async, bounded concurrency)
# ---------------------------------------------------------------------------

async def grade_batch(
    traces: list[dict],
    concurrency: int = 8,
    output_path: "str | Path | None" = None,
) -> list[dict]:
    """Grade a list of inference traces with bounded async concurrency.

    Uses an async semaphore to cap parallel Anthropic calls at ``concurrency``.
    The programmatic grader runs inline (no I/O), and the LLM judge is called
    via ``asyncio.to_thread`` so it doesn't block the event loop.

    Parameters
    ----------
    traces:
        List of inference trace dicts.
    concurrency:
        Max simultaneous Anthropic judge calls. Default 8.
    output_path:
        Optional path to append graded records to as they complete.
        When provided, each record is written immediately upon completion
        (append mode, flushed) and a tqdm progress bar is shown.
        When None (default), behaves as before — no bar, no streaming.

    Returns
    -------
    list[dict]
        Graded records (order matches completion order when output_path is
        given, otherwise matches input order).
    """
    sem = asyncio.Semaphore(concurrency)
    # One shared sync client; thread-safe for reads.
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    async def _grade_one_async(trace: dict) -> dict:
        async with sem:
            return await asyncio.to_thread(grade_one, trace, client)

    if output_path is None:
        # Legacy path: no bar, no streaming, order preserved.
        return list(await asyncio.gather(*[_grade_one_async(t) for t in traces]))

    # Streaming path: tqdm bar + append-as-you-go.
    from tqdm.asyncio import tqdm as tqdm_asyncio  # noqa: PLC0415

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = [_grade_one_async(t) for t in traces]
    results: list[dict] = []

    with open(output_path, "a", encoding="utf-8") as fh:
        for coro in tqdm_asyncio.as_completed(
            tasks,
            total=len(tasks),
            desc="Grading",
            unit="q",
        ):
            graded = await coro
            results.append(graded)
            fh.write(json.dumps(graded, ensure_ascii=False) + "\n")
            fh.flush()

    return results
