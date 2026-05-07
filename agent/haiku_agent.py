"""Haiku agent inference loop.

Single tool (``search_wikipedia``), max 5 tool calls per turn, hard stop
after which the model is asked once more for a final answer with the tool
*removed* from the spec — this guarantees termination and a final natural
language answer even if the model keeps wanting to search.

Returns an :data:`InferenceTrace` dict matching the ``runs/<ts>.inference.jsonl``
contract in ``PLANS.md``. Tool description and system prompt IDs are echoed
into the trace so downstream eval / McNemar's can attribute.

CLI smoke (real APIs):
    python -m agent.haiku_agent "When did the Severance season 2 finale air?"

Tests mock the Anthropic SDK and the tool — see ``tests/test_haiku_agent.py``.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from agent.prompts import SYSTEM_PROMPTS, TOOL_DESCRIPTIONS
from wiki.client import search_wikipedia as _real_search_wikipedia

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "claude-haiku-4-5"

# Hard cap per PLANS.md.
MAX_TOOL_CALLS = 10

# Anthropic SDK requires max_tokens; pick something generous but bounded.
MAX_TOKENS = 4096

# Locked input_schema. Description text varies; schema does not.
TOOL_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
}

TOOL_NAME = "search_wikipedia"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_dotenv_if_present() -> None:
    """Best-effort .env load — don't hard-require python-dotenv."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    try:
        from dotenv import load_dotenv  # type: ignore
    except ImportError:
        # Hand-roll a minimal parser so the smoke CLI works without dotenv.
        env_path = Path(__file__).resolve().parent.parent / ".env"
        if env_path.exists():
            for raw in env_path.read_text().splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'").strip('"'))
        return
    load_dotenv()


def _make_client() -> Any:
    """Construct an Anthropic SDK client. Imported lazily so tests can monkey-patch."""
    _load_dotenv_if_present()
    import anthropic  # type: ignore

    return anthropic.Anthropic()


def _build_tool_param(tool_description_id: str) -> dict[str, Any]:
    if tool_description_id not in TOOL_DESCRIPTIONS:
        raise KeyError(
            f"unknown tool_description_id={tool_description_id!r} "
            f"(available: {sorted(TOOL_DESCRIPTIONS)})"
        )
    return {
        "name": TOOL_NAME,
        "description": TOOL_DESCRIPTIONS[tool_description_id],
        "input_schema": TOOL_INPUT_SCHEMA,
    }


def _render_system_prompt(raw: str) -> str:
    """Substitute placeholders in a system-prompt template.

    Currently supports ``{{TIMESTAMP}}`` → today's date in ISO form
    (``YYYY-MM-DD`` UTC). Add more placeholders here if/when needed.
    """
    if "{{TIMESTAMP}}" in raw:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        raw = raw.replace("{{TIMESTAMP}}", today)
    return raw


def _resolve_system_prompt(system_prompt_id: str) -> str:
    if system_prompt_id not in SYSTEM_PROMPTS:
        raise KeyError(
            f"unknown system_prompt_id={system_prompt_id!r} "
            f"(available: {sorted(SYSTEM_PROMPTS)})"
        )
    return _render_system_prompt(SYSTEM_PROMPTS[system_prompt_id])


def _content_to_dicts(content: Any) -> list[dict[str, Any]]:
    """Normalize Anthropic response content blocks to plain dicts.

    The SDK returns rich objects (TextBlock, ToolUseBlock); we serialize them
    so the trace can be JSON-dumped to disk verbatim.
    """
    out: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, dict):
            out.append(block)
            continue
        block_type = getattr(block, "type", None)
        if block_type == "text":
            out.append({"type": "text", "text": getattr(block, "text", "")})
        elif block_type == "tool_use":
            out.append(
                {
                    "type": "tool_use",
                    "id": getattr(block, "id", None),
                    "name": getattr(block, "name", None),
                    "input": getattr(block, "input", {}) or {},
                }
            )
        elif block_type == "thinking":
            # Future-proof: extended thinking blocks. Pass through opaquely.
            out.append(
                {
                    "type": "thinking",
                    "thinking": getattr(block, "thinking", ""),
                }
            )
        else:
            # Unknown block — best-effort dict conversion.
            try:
                out.append(dict(block))  # type: ignore[arg-type]
            except Exception:  # noqa: BLE001
                out.append({"type": str(block_type), "raw": repr(block)})
    return out


def _extract_final_text(content_dicts: list[dict[str, Any]]) -> str:
    parts = [b.get("text", "") for b in content_dicts if b.get("type") == "text"]
    return "\n".join(p for p in parts if p).strip()


def _tool_use_blocks(content_dicts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [b for b in content_dicts if b.get("type") == "tool_use"]


def _run_tool(
    tool_use_block: dict[str, Any],
    tool_fn: Callable[[str], dict[str, Any]],
) -> dict[str, Any]:
    """Execute one tool call. Always returns a tool_result content block."""
    tool_use_id = tool_use_block.get("id")
    name = tool_use_block.get("name")
    if name != TOOL_NAME:
        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": json.dumps({"error": f"unknown tool: {name}"}),
            "is_error": True,
        }
    inp = tool_use_block.get("input") or {}
    query = inp.get("query", "")
    try:
        result = tool_fn(query)
    except Exception as e:  # noqa: BLE001 — defensive; the real tool never raises
        result = {"title": None, "url": None, "lead": "", "error": f"tool_exc: {e}"}
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": json.dumps(result),
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_agent(
    question: str,
    tool_description_id: str,
    system_prompt_id: str = "baseline",
    model: str = DEFAULT_MODEL,
    *,
    client: Any | None = None,
    tool_fn: Callable[[str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run the Haiku agent on ``question`` and return an inference trace.

    Args:
        question: User question.
        tool_description_id: Key into :data:`TOOL_DESCRIPTIONS`.
        system_prompt_id: Key into :data:`SYSTEM_PROMPTS`. Defaults to
            ``"baseline"`` (empty string).
        model: Anthropic model id. Defaults to ``claude-haiku-4-5``.
        client: Pre-built Anthropic client (tests inject a mock).
        tool_fn: Override the tool function (tests inject a stub).

    Returns:
        Inference trace dict matching ``PLANS.md`` contract.
    """
    tool_param = _build_tool_param(tool_description_id)
    system_prompt = _resolve_system_prompt(system_prompt_id)
    if tool_fn is None:
        tool_fn = _real_search_wikipedia
    if client is None:
        client = _make_client()

    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
    n_tool_calls = 0
    final_answer = ""
    t_start = time.perf_counter()

    while True:
        # Build kwargs for messages.create. Tools are removed once we hit the
        # hard cap, forcing a final natural-language answer.
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": MAX_TOKENS,
            "messages": messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if n_tool_calls < MAX_TOOL_CALLS:
            kwargs["tools"] = [tool_param]

        response = client.messages.create(**kwargs)
        content_dicts = _content_to_dicts(response.content)
        # Record assistant turn into the trace.
        messages.append({"role": "assistant", "content": content_dicts})

        stop_reason = getattr(response, "stop_reason", None)
        tool_uses = _tool_use_blocks(content_dicts)

        # Termination paths --------------------------------------------------
        if not tool_uses:
            # Either end_turn or stop_sequence — we have the final answer.
            final_answer = _extract_final_text(content_dicts)
            break

        # Cap reached *before* this round — already handled above by removing
        # the tools param. If the model still emitted tool_use under no tools
        # (shouldn't happen), treat its text as final and bail.
        if "tools" not in kwargs:
            final_answer = _extract_final_text(content_dicts)
            break

        # Run tool calls and feed results back -------------------------------
        # If the model emitted multiple tool_use blocks in one turn (rare but
        # legal), we resolve all of them in one user message.
        tool_results: list[dict[str, Any]] = []
        for block in tool_uses:
            tool_results.append(_run_tool(block, tool_fn))
            n_tool_calls += 1

        messages.append({"role": "user", "content": tool_results})

        # If the cap was *just* reached, do one more loop pass with tools
        # stripped so the model produces a final answer instead of hanging.
        if n_tool_calls >= MAX_TOOL_CALLS:
            kwargs2: dict[str, Any] = {
                "model": model,
                "max_tokens": MAX_TOKENS,
                "messages": messages,
            }
            if system_prompt:
                kwargs2["system"] = system_prompt
            response2 = client.messages.create(**kwargs2)
            content2 = _content_to_dicts(response2.content)
            messages.append({"role": "assistant", "content": content2})
            final_answer = _extract_final_text(content2)
            stop_reason = getattr(response2, "stop_reason", stop_reason)
            break

        # Otherwise loop and let the model continue.

    latency_ms = int((time.perf_counter() - t_start) * 1000)

    return {
        # question_id and tool_required_label are filled in by run.py — the
        # agent doesn't know them. We still emit the rest of the contract.
        "user_question": question,
        "messages": messages,
        "final_answer": final_answer,
        "n_tool_calls": n_tool_calls,
        "tool_description_id": tool_description_id,
        "system_prompt_id": system_prompt_id,
        "model": model,
        "latency_ms": latency_ms,
    }


# ---------------------------------------------------------------------------
# CLI smoke (real APIs — gated to __main__ so pytest never triggers it)
# ---------------------------------------------------------------------------


def _smoke_main(argv: list[str]) -> int:  # pragma: no cover
    if not argv:
        print(
            "usage: python -m agent.haiku_agent <question> "
            "[--tool-description-id baseline] [--system-prompt-id baseline]",
            file=sys.stderr,
        )
        return 2
    # Tiny inline arg parse to avoid argparse boilerplate noise.
    question_parts: list[str] = []
    tool_id = "baseline"
    sys_id = "baseline"
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--tool-description-id" and i + 1 < len(argv):
            tool_id = argv[i + 1]
            i += 2
        elif a == "--system-prompt-id" and i + 1 < len(argv):
            sys_id = argv[i + 1]
            i += 2
        else:
            question_parts.append(a)
            i += 1
    question = " ".join(question_parts).strip()
    if not question:
        print("error: empty question", file=sys.stderr)
        return 2
    trace = run_agent(question, tool_description_id=tool_id, system_prompt_id=sys_id)
    # Pretty-print a reviewer-friendly summary, then dump the full trace.
    print("=" * 60)
    print(f"question:           {question}")
    print(f"tool_description:   {tool_id}")
    print(f"system_prompt:      {sys_id}")
    print(f"n_tool_calls:       {trace['n_tool_calls']}")
    print(f"latency_ms:         {trace['latency_ms']}")
    print("-" * 60)
    print(f"final_answer:\n{trace['final_answer']}")
    print("=" * 60)
    print("full trace (JSON):")
    print(json.dumps(trace, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_smoke_main(sys.argv[1:]))
