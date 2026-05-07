"""Reviewer-facing demo CLI.

Two modes:

    python demo.py query
        Stratified random sample of 3 questions (one tool-not-required, one
        tool-required TV, one tool-required sports). Runs the agent on each,
        pretty-prints the conversation. Use --question "..." for a custom
        one-off.

    python demo.py eval --limit 20
        Stratified random subsample of the eval set. Runs the full agent +
        grader pipeline, writes runs/<ts>__td-X__sp-Y/, then prints the
        granular accuracy breakdown as a stdout table.

Both subcommands accept --tool-description-id and --system-prompt-id so a
reviewer can flip between baseline and v1 to see prompt iteration in action.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from agent.haiku_agent import run_agent
from eval.grade import grade_batch
from eval.stats import compute_metrics
from run import RUNS_DIR, load_dataset, run_inference


# ---------------------------------------------------------------------------
# Color helpers (no-op when stdout isn't a TTY, e.g. piped to a file)
# ---------------------------------------------------------------------------

_COLOR = sys.stdout.isatty()


class C:
    RESET = "\033[0m" if _COLOR else ""
    BOLD = "\033[1m" if _COLOR else ""
    DIM = "\033[2m" if _COLOR else ""
    CYAN = "\033[36m" if _COLOR else ""
    MAGENTA = "\033[35m" if _COLOR else ""
    GREEN = "\033[32m" if _COLOR else ""
    YELLOW = "\033[33m" if _COLOR else ""
    RED = "\033[31m" if _COLOR else ""


# ---------------------------------------------------------------------------
# Trace pretty-printing
# ---------------------------------------------------------------------------

def _truncate(s: str, n: int = 200) -> str:
    s = " ".join(s.split())  # collapse all whitespace
    return s if len(s) <= n else s[: n - 1] + "…"


def _try_json(s: Any) -> Any:
    if not isinstance(s, str):
        return None
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return None


def _pretty_print_trace(trace: dict, header: str) -> None:
    print()
    print(f"{C.BOLD}{C.CYAN}━━━ {header} ━━━{C.RESET}")
    print(f"{C.BOLD}Q:{C.RESET} {trace.get('user_question', '?')}")
    print()

    for msg in trace.get("messages") or []:
        content = msg.get("content")
        if not isinstance(content, list):
            continue

        if msg.get("role") == "assistant":
            for block in content:
                if block.get("type") == "tool_use":
                    query = (block.get("input") or {}).get("query", "?")
                    print(f"  {C.MAGENTA}🔍 search_wikipedia({query!r}){C.RESET}")

        elif msg.get("role") == "user":
            for block in content:
                if block.get("type") != "tool_result":
                    continue
                content_str = block.get("content", "")
                parsed = _try_json(content_str)
                if parsed and isinstance(parsed, dict) and parsed.get("results"):
                    top = parsed["results"][0]
                    title = top.get("title", "?")
                    extract = _truncate(top.get("extract") or "", 200)
                    print(f"     {C.DIM}↳ {title}{C.RESET}")
                    print(f"       {C.DIM}{extract}{C.RESET}")
                else:
                    print(f"     {C.DIM}↳ {_truncate(str(content_str), 200)}{C.RESET}")

    print()
    print(f"{C.BOLD}A:{C.RESET} {trace.get('final_answer', '')}")
    latency_s = trace.get("latency_ms", 0) / 1000
    n_calls = trace.get("n_tool_calls", 0)
    print(f"{C.DIM}⏱ {latency_s:.1f}s · 🔧 {n_calls} tool calls{C.RESET}")


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _sample_query_set(examples: list[dict], rng: random.Random) -> list[dict]:
    """One random example each from {tool_not_required, tool_required+tv,
    tool_required+sports}. Skips strata with no examples."""
    by_slice: dict[str, list[dict]] = {}
    for ex in examples:
        if not ex.get("tool_required_label"):
            key = "tool_not_required"
        elif ex.get("category") == "sports":
            key = "tool_required_sports"
        elif ex.get("category") == "tv":
            key = "tool_required_tv"
        else:
            key = "tool_required_other"
        by_slice.setdefault(key, []).append(ex)

    picked = []
    for key in ("tool_not_required", "tool_required_tv", "tool_required_sports"):
        if by_slice.get(key):
            picked.append(rng.choice(by_slice[key]))
    return picked


def _stratified_subsample(
    examples: list[dict], n: int, rng: random.Random
) -> list[dict]:
    """Stratified random subsample preserving dataset proportions across
    (tool_required_label, category)."""
    strata: dict[tuple, list[dict]] = {}
    for ex in examples:
        key = (bool(ex.get("tool_required_label")), ex.get("category"))
        strata.setdefault(key, []).append(ex)

    total = len(examples)
    sampled: list[dict] = []
    for group in strata.values():
        share = round(n * len(group) / total)
        share = max(1, min(share, len(group)))
        sampled.extend(rng.sample(group, share))

    rng.shuffle(sampled)
    if len(sampled) > n:
        return sampled[:n]
    if len(sampled) < n:
        leftover = [ex for ex in examples if ex not in sampled]
        rng.shuffle(leftover)
        sampled.extend(leftover[: n - len(sampled)])
    return sampled


# ---------------------------------------------------------------------------
# Subcommand: query
# ---------------------------------------------------------------------------

def _slice_label(ex: dict) -> str:
    if not ex.get("tool_required_label"):
        return "tool-not-required (stable)"
    return f"tool-required: {ex.get('category', '?')}"


def cmd_query(args: argparse.Namespace) -> None:
    load_dotenv()
    print(
        f"{C.DIM}tool_description_id={args.tool_description_id}  "
        f"system_prompt_id={args.system_prompt_id}{C.RESET}"
    )

    if args.question:
        trace = run_agent(
            args.question,
            args.tool_description_id,
            system_prompt_id=args.system_prompt_id,
        )
        _pretty_print_trace(trace, "custom query")
        return

    rng = random.Random(args.seed) if args.seed is not None else random.Random()
    examples = load_dataset()
    sample = _sample_query_set(examples, rng)
    if not sample:
        print(f"{C.RED}No examples available — run datagen first.{C.RESET}")
        sys.exit(1)

    for i, ex in enumerate(sample, 1):
        header = f"Q{i}/{len(sample)} — {_slice_label(ex)}"
        trace = run_agent(
            ex["user_question"],
            args.tool_description_id,
            system_prompt_id=args.system_prompt_id,
        )
        _pretty_print_trace(trace, header)
        print(f"{C.GREEN}{C.DIM}golden:{C.RESET} {ex.get('golden_answer', '')}")


# ---------------------------------------------------------------------------
# Subcommand: eval
# ---------------------------------------------------------------------------

def _format_pct(x: float) -> str:
    return f"{x*100:5.1f}%"


def _row(label: str, slice_dict: dict, indent: int = 0) -> str:
    pad = "  " * indent
    n = slice_dict.get("n", 0)
    nc = slice_dict.get("n_correct", 0)
    acc = slice_dict.get("accuracy", 0.0)
    return f"{pad}{label:<30} {n:>5} {nc:>7} {_format_pct(acc):>8}"


def _print_summary_table(summary: dict) -> None:
    ac = summary["answer_correctness"]
    tnr = summary["tool_not_required"]

    print()
    print(f"{C.BOLD}Answer correctness on tool-required:{C.RESET}")
    print(f"  {'subset':<30} {'n':>5} {'correct':>7} {'%':>8}")
    print(f"  {'-'*30} {'-'*5} {'-'*7} {'-'*8}")
    print("  " + _row("overall", ac))
    btc = ac.get("by_tool_called", {})
    if btc.get("true", {}).get("n"):
        print("  " + _row("called tool", btc["true"], indent=1))
    if btc.get("false", {}).get("n"):
        print("  " + _row("didn't call", btc["false"], indent=1))
    for cat, cat_slice in (ac.get("by_category") or {}).items():
        print("  " + _row(cat, cat_slice, indent=1))

    print()
    n_called = ac.get("n_tool_called", 0)
    pct = ac.get("pct_tool_called", 0.0)
    print(
        f"{C.BOLD}Tool-call rate (tool-required):{C.RESET} "
        f"{n_called}/{ac['n']} ({_format_pct(pct).strip()})"
    )
    print(
        f"{C.BOLD}Tool-call rate (tool-not-required):{C.RESET} "
        f"{tnr['n_tool_called']}/{tnr['n']} ({_format_pct(tnr['pct_tool_called']).strip()})"
    )

    tu = summary["tool_use"]
    print()
    print(
        f"{C.BOLD}Tool-use confusion (full set):{C.RESET} "
        f"acc={_format_pct(tu['accuracy']).strip()}  "
        f"prec={tu['precision']:.3f}  "
        f"recall={tu['recall']:.3f}  "
        f"(tp={tu['tp']} fp={tu['fp']} tn={tu['tn']} fn={tu['fn']})"
    )
    print(f"{C.BOLD}Avg tool calls/q:{C.RESET} {summary['n_tool_calls_avg']:.2f}")


async def _run_eval(args: argparse.Namespace) -> None:
    load_dotenv()

    rng = random.Random(args.seed) if args.seed is not None else random.Random()
    examples = load_dataset()
    sample = _stratified_subsample(examples, args.limit, rng)

    n_req = sum(1 for ex in sample if ex["tool_required_label"])
    print(
        f"Loaded {len(sample)} examples "
        f"(stratified subsample of {len(examples)}: "
        f"{n_req} tool-required, {len(sample) - n_req} tool-not-required)"
    )

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = (
        RUNS_DIR
        / f"{run_id}__td-{args.tool_description_id}__sp-{args.system_prompt_id}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    inference_path = run_dir / "inference.jsonl"
    results_path = run_dir / "results.jsonl"
    summary_path = run_dir / "summary.json"

    print(
        f"Run {run_id}: "
        f"tool_description_id={args.tool_description_id} "
        f"system_prompt_id={args.system_prompt_id}"
    )

    t0 = time.time()
    traces = await run_inference(
        sample,
        args.tool_description_id,
        args.system_prompt_id,
        args.concurrency,
        output_path=inference_path,
    )
    print(f"Inference: {len(traces)} traces in {time.time() - t0:.1f}s")

    t0 = time.time()
    graded = await grade_batch(traces, output_path=results_path)
    print(f"Grading: {len(graded)} graded in {time.time() - t0:.1f}s")

    summary = compute_metrics(graded)
    summary["run_id"] = run_id
    summary["tool_description_id"] = args.tool_description_id
    summary["system_prompt_id"] = args.system_prompt_id
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    _print_summary_table(summary)
    print()
    print(f"{C.DIM}Run dir: {run_dir}{C.RESET}")


def cmd_eval(args: argparse.Namespace) -> None:
    asyncio.run(_run_eval(args))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reviewer-facing demo CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("query", help="Run agent on 3 sampled questions (or one custom)")
    q.add_argument("--question", default=None, help="Custom question (skips sampling)")
    q.add_argument("--tool-description-id", default="baseline")
    q.add_argument("--system-prompt-id", default="baseline")
    q.add_argument("--seed", type=int, default=None, help="Optional random seed")
    q.set_defaults(func=cmd_query)

    e = sub.add_parser("eval", help="Run subset eval and print breakdown")
    e.add_argument("--limit", type=int, default=20)
    e.add_argument("--concurrency", type=int, default=8)
    e.add_argument("--tool-description-id", default="baseline")
    e.add_argument("--system-prompt-id", default="baseline")
    e.add_argument("--seed", type=int, default=None, help="Optional random seed")
    e.set_defaults(func=cmd_eval)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.func(args)
