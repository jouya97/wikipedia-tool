"""
datagen/gen_tool_required.py

Generate tool_required.jsonl: 384 synthetic Q&A pairs grounded in post-March-2025
Wikipedia facts.  24 async Opus-4.7 (medium-effort) workers each handle ~16 seeds
sequentially inside one growing conversation.

CLI:
    python -m datagen.gen_tool_required --seeds data/seeds.jsonl \
        --out data/tool_required.jsonl --target 384

Smoke test:
    python -m datagen.gen_tool_required --seeds data/seeds.jsonl \
        --out data/tool_required_smoke.jsonl --target 4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
# Quiet by default: only warnings/errors surface alongside the tqdm bar.
# Set DATAGEN_LOG_LEVEL=INFO (or DEBUG) to re-enable verbose per-seed chatter.
_log_level = os.environ.get("DATAGEN_LOG_LEVEL", "WARNING").upper()
logging.basicConfig(level=_log_level, format="%(levelname)s %(message)s")
# Silence noisy HTTP-layer loggers regardless of level.
for _noisy in ("httpx", "httpcore", "anthropic", "urllib3"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL = "claude-opus-4-7"
# Per Anthropic docs: effort controls thinking depth and total token spend on Opus 4.7.
# "medium" is the recommended drop-in for cost-aware agentic workflows.
EFFORT = "medium"
MAX_TOKENS = 8192
NUM_WORKERS = 24
MAX_TOOL_CALLS_PER_SEED = 5
# Default Q&As per seed — overridable via --q-per-seed. With 141 seeds × 5 Q&As
# = 705 raw, ~470 after Opus self-trimming, dedup'd and trimmed to TARGET.
Q_AS_PER_SEED = 5
DEDUP_JACCARD_THRESHOLD = 0.5  # 5-gram Jaccard above this → duplicate
SEEDS_FILE = "data/seeds.jsonl"
OUT_FILE = "data/tool_required.jsonl"
TARGET = 384

# ---------------------------------------------------------------------------
# Tool spec for Opus (mirrors the contract in PLANS.md)
# ---------------------------------------------------------------------------

SEARCH_TOOL_SPEC: dict[str, Any] = {
    "name": "search_wikipedia",
    "description": "Search Wikipedia for an article matching the query.",
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
}

# ---------------------------------------------------------------------------
# System prompt for data-gen workers
# ---------------------------------------------------------------------------

def _build_system_prompt(q_per_seed: int) -> str:
    """Build the datagen system prompt for a given per-seed Q&A budget."""
    return f"""\
You are a high-quality synthetic-data generator for an AI evaluation benchmark.
Your job is to use the search_wikipedia tool to ground factual answers, then
produce SYNTHETIC Q&A PAIRS that a curious human might send to a Wikipedia-aware
AI assistant.

For each seed topic you will:
1. Call search_wikipedia (up to 5 times) to find relevant facts about the topic,
   focusing especially on developments AFTER March 2025.
2. Based on what you found, compose a JSON ARRAY of UP TO {q_per_seed} distinct
   Q&A objects, each on a DIFFERENT angle of what the article says (examples of
   distinct angles: airdates / season finale; cast or roster changes; plot or
   results; reception, ratings, or awards; comparisons to prior seasons; behind-
   the-scenes news; renewals/cancellations). Output ONLY the JSON array — no
   prose, no markdown fences. Each object MUST use this schema:
   {{
     "seed_title": "<the topic title you were given>",
     "user_question": "<colloquial question — see framing rules below>",
     "golden_answer": "<2-4 sentence factual answer grounded in the search results>"
   }}
   If you can only confidently ground 1 or 2 distinct angles, return that many —
   DO NOT fabricate to hit the count. Quality over quantity.
3. If you cannot ground ANY factual answer (no relevant article, or lead section
   lacks post-March-2025 facts), output ONLY this JSON OBJECT (not an array):
   {{"seed_title": "<title>", "skip_reason": "<one sentence reason>"}}

USER-QUESTION FRAMING RULES (follow verbatim):
Phrase user_question like an actual curious person typing it, NOT like a search
query. Vary register: casual ("yo did X really happen?"), context-setting ("I heard
last month that..."), opinion-bait ("is X overrated now?"), follow-up-shaped
("wait, didn't they replace Y?"), comparative ("how does the new season compare to
the old one?"). Do NOT phrase like "When did season 3 of X premiere?" — that's
search-engine-shaped and trivially retrievable. The question must still be fully
answerable from the searches you ran.

Additional quality rules:
- The golden_answer must be factual and grounded in the Wikipedia lead you retrieved.
- Do not fabricate facts not present in the search results.
- user_question must be a genuine question (ends with "?").
- Output ONLY the JSON object — no markdown fences, no commentary.
- User questions MUST refer to the seed title so the AI assistant can actually form a query without asking for further clarification.
"""


# Module-level default prompt, kept for back-compat with any external importers.
DATAGEN_SYSTEM_PROMPT = _build_system_prompt(Q_AS_PER_SEED)


# ---------------------------------------------------------------------------
# 5-gram Jaccard near-duplicate detection
# ---------------------------------------------------------------------------

def _ngrams(text: str, n: int = 5) -> set[str]:
    """Return a set of character n-grams from normalised text."""
    t = re.sub(r"\s+", " ", text.lower().strip())
    if len(t) < n:
        return {t}
    return {t[i : i + n] for i in range(len(t) - n + 1)}


def _jaccard(a: str, b: str, n: int = 5) -> float:
    sa, sb = _ngrams(a, n), _ngrams(b, n)
    if not sa and not sb:
        return 1.0
    intersection = len(sa & sb)
    union = len(sa | sb)
    return intersection / union if union else 0.0


def dedup_records(records: list[dict]) -> list[dict]:
    """
    Remove near-duplicate records based on user_question 5-gram Jaccard.
    Keeps the first occurrence; drops later ones above the threshold.
    """
    kept: list[dict] = []
    for rec in records:
        q = rec.get("user_question", "")
        is_dup = any(
            _jaccard(q, k.get("user_question", "")) > DEDUP_JACCARD_THRESHOLD
            for k in kept
        )
        if not is_dup:
            kept.append(rec)
    return kept


# ---------------------------------------------------------------------------
# Seed loading and partitioning
# ---------------------------------------------------------------------------

def load_seeds(path: str) -> list[dict]:
    seeds = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                seeds.append(json.loads(line))
    return seeds


def partition_seeds(seeds: list[dict], n_workers: int = NUM_WORKERS) -> list[list[dict]]:
    """Partition seeds by topic_slice (0..n_workers-1)."""
    buckets: dict[int, list[dict]] = defaultdict(list)
    for s in seeds:
        slice_id = int(s.get("topic_slice", 0)) % n_workers
        buckets[slice_id].append(s)
    return [buckets[i] for i in range(n_workers)]


# ---------------------------------------------------------------------------
# Execute a single search_wikipedia tool call (lazy import)
# ---------------------------------------------------------------------------

def _call_search_wikipedia(query: str) -> dict:
    """Lazy-import search_wikipedia from wiki.client and call it."""
    from wiki.client import search_wikipedia  # noqa: PLC0415
    return search_wikipedia(query)


# ---------------------------------------------------------------------------
# Single worker: processes a list of seeds sequentially in one conversation
# ---------------------------------------------------------------------------

async def _worker(
    worker_id: int,
    seeds: list[dict],
    client: Any,
    progress_bar: Any = None,
    partial_fh: Any = None,
    q_per_seed: int = Q_AS_PER_SEED,
) -> tuple[list[dict], list[dict]]:
    """
    Process `seeds` sequentially in one growing Opus conversation.

    Returns (successes, skips):
      - successes: list of valid tool_required records
      - skips:     list of {"seed_title": ..., "skip_reason": ...}

    Parameters
    ----------
    worker_id:
        Numeric ID for logging.
    seeds:
        Seeds assigned to this worker.
    client:
        Anthropic sync client.
    progress_bar:
        Optional shared tqdm instance; advanced by 1 after each seed.
    partial_fh:
        Optional open file handle (append mode) for partial-save JSONL.
        Each processed seed appends one line with a ``_kind`` marker.
    """
    successes: list[dict] = []
    skips: list[dict] = []

    system_prompt = _build_system_prompt(q_per_seed)

    # Growing conversation — each seed appends to messages
    messages: list[dict] = []

    for seed in seeds:
        title = seed.get("title", "")
        logger.debug("Worker %d processing seed: %s", worker_id, title)

        # Add a new user turn for this seed
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Next seed topic: {title}\n"
                    f"(post-March-2025 hint: {seed.get('post_march_2025_fact', '')})\n\n"
                    f"Use search_wikipedia to ground factual answers, then output a "
                    f"JSON ARRAY of up to {q_per_seed} distinct Q&A objects (different "
                    "angles of the article). Output ONLY the JSON array — no prose, "
                    "no markdown fences."
                ),
            }
        )

        tool_call_count = 0
        search_queries: list[str] = []

        # Agentic loop for this seed
        while True:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda m=list(messages): client.messages.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    system=system_prompt,
                    tools=[SEARCH_TOOL_SPEC],
                    messages=m,
                    output_config={"effort": EFFORT},
                ),
            )

            # Append assistant response to conversation
            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            # Check for tool use blocks
            tool_use_blocks = [b for b in assistant_content if b.type == "tool_use"]

            if not tool_use_blocks:
                # Final answer — extract JSON from text blocks
                text_blocks = [b for b in assistant_content if b.type == "text"]
                raw_text = " ".join(b.text for b in text_blocks).strip()
                parsed = _extract_json(raw_text)
                seed_results: list[dict] = []
                seed_skips: list[dict] = []
                if parsed is None:
                    seed_skips.append({"seed_title": title, "skip_reason": "Could not parse JSON from response"})
                elif isinstance(parsed, dict) and "skip_reason" in parsed:
                    seed_skips.append(parsed)
                elif isinstance(parsed, list):
                    # Multi-record output: validate each, share search provenance.
                    n_added = 0
                    for rec in parsed:
                        if isinstance(rec, dict) and _validate_record(rec):
                            rec["datagen_search_queries"] = list(search_queries)
                            rec["datagen_n_tool_calls"] = tool_call_count
                            seed_results.append(rec)
                            n_added += 1
                    if n_added == 0:
                        seed_skips.append({"seed_title": title, "skip_reason": "No valid records in array"})
                elif isinstance(parsed, dict) and _validate_record(parsed):
                    # Back-compat: single object output.
                    parsed["datagen_search_queries"] = search_queries
                    parsed["datagen_n_tool_calls"] = tool_call_count
                    seed_results.append(parsed)
                else:
                    seed_skips.append({"seed_title": title, "skip_reason": "Missing required fields in output"})

                successes.extend(seed_results)
                skips.extend(seed_skips)

                # Partial-save: write each outcome immediately.
                if partial_fh is not None:
                    for rec in seed_results:
                        partial_fh.write(
                            json.dumps({**rec, "_kind": "success"}, ensure_ascii=False) + "\n"
                        )
                    for skip in seed_skips:
                        partial_fh.write(
                            json.dumps({**skip, "_kind": "skip"}, ensure_ascii=False) + "\n"
                        )
                    partial_fh.flush()

                # Advance shared progress bar.
                if progress_bar is not None:
                    progress_bar.update(1)

                break

            # Hard cap on tool calls
            if tool_call_count >= MAX_TOOL_CALLS_PER_SEED:
                # Tell model to finalize without more tool use
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "You have reached the maximum number of tool calls. "
                            "Please output the final JSON now based on what you have found."
                        ),
                    }
                )
                continue

            # Execute all tool calls
            tool_results = []
            for tb in tool_use_blocks:
                if tb.name == "search_wikipedia":
                    query = tb.input.get("query", "")
                    search_queries.append(query)
                    tool_call_count += 1
                    loop = asyncio.get_running_loop()
                    result = await loop.run_in_executor(
                        None, _call_search_wikipedia, query
                    )
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tb.id,
                            "content": json.dumps(result),
                        }
                    )

            messages.append({"role": "user", "content": tool_results})

        # End of seed loop

    return successes, skips


def _extract_json(text: str) -> dict | None:
    """Extract the first valid JSON object from text."""
    # Strip markdown fences if present
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`")
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to find a JSON object in the text
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def _validate_record(rec: dict) -> bool:
    required = {"seed_title", "user_question", "golden_answer"}
    return required.issubset(rec.keys()) and rec.get("user_question", "").strip().endswith("?")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

async def generate(
    seeds_path: str,
    out_path: str,
    target: int,
    client: Any,
    q_per_seed: int = Q_AS_PER_SEED,
) -> list[dict]:
    """
    Run 24 workers over the seeds, retry until `target` successes after dedup.
    """
    from tqdm import tqdm  # noqa: PLC0415

    seeds = load_seeds(seeds_path)
    tqdm.write(f"Loaded {len(seeds)} seeds from {seeds_path}")

    partial_path = Path(out_path).with_suffix(".partial.jsonl")
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    # Truncate/create the partial file at the start of each generate() run.
    partial_path.write_text("")

    # Keep a held-back pool for replacements (seeds beyond the primary 24*16=384)
    primary_seeds = seeds[:target]
    replacement_pool = list(seeds[target:])  # anything beyond target

    all_successes: list[dict] = []
    attempt = 0

    # Phase 1: gather raw successes (retry on skips)
    with open(partial_path, "a", encoding="utf-8") as partial_fh:
        while len(all_successes) < target:
            attempt += 1
            tqdm.write(f"Attempt {attempt} — need {target - len(all_successes)} more raw successes")

            # Partition available seeds
            available = primary_seeds if attempt == 1 else replacement_pool[: target - len(all_successes)]
            if not available:
                logger.warning("Ran out of replacement seeds at %d successes — stopping.", len(all_successes))
                break

            partitions = partition_seeds(available, NUM_WORKERS)

            # One shared tqdm bar across all workers for this attempt.
            total_seeds_this_attempt = sum(len(p) for p in partitions if p)
            pbar = tqdm(
                total=total_seeds_this_attempt,
                desc=f"Datagen attempt {attempt}",
                unit="seed",
            )

            tasks = [
                _worker(i, partitions[i], client, progress_bar=pbar, partial_fh=partial_fh, q_per_seed=q_per_seed)
                for i in range(NUM_WORKERS)
                if partitions[i]
            ]
            results = await asyncio.gather(*tasks)
            pbar.close()

            new_successes = []
            new_skips = []
            for successes, skips in results:
                new_successes.extend(successes)
                new_skips.extend(skips)

            tqdm.write(
                f"Attempt {attempt}: {len(new_successes)} successes, {len(new_skips)} skips"
            )
            all_successes.extend(new_successes)

            # Replenish replacement pool entries for each skip
            for _skip in new_skips:
                if replacement_pool:
                    replacement_pool.pop(0)

            if attempt > 5:
                logger.warning("Exceeded 5 retry attempts — using what we have.")
                break

        # Phase 2: near-duplicate filter, then retry if below target
        pre_dedup = len(all_successes)
        unique = dedup_records(all_successes)
        tqdm.write(f"Dedup: {pre_dedup} → {len(unique)} records")

        if len(unique) < target and replacement_pool:
            tqdm.write(
                f"Dedup dropped to {len(unique)} — retrying with "
                f"{len(replacement_pool)} replacement seeds"
            )
            # One extra pass with remaining replacement seeds
            needed = target - len(unique)
            extra_seeds = replacement_pool[:needed]
            if extra_seeds:
                partitions = partition_seeds(extra_seeds, NUM_WORKERS)
                extra_total = sum(len(p) for p in partitions if p)
                pbar = tqdm(
                    total=extra_total,
                    desc="Datagen dedup-retry",
                    unit="seed",
                )
                tasks = [
                    _worker(i, partitions[i], client, progress_bar=pbar, partial_fh=partial_fh, q_per_seed=q_per_seed)
                    for i in range(NUM_WORKERS)
                    if partitions[i]
                ]
                results = await asyncio.gather(*tasks)
                pbar.close()
                extra_successes = [s for successes, _ in results for s in successes]
                all_successes.extend(extra_successes)
                unique = dedup_records(all_successes)
                tqdm.write(f"After extra dedup pass: {len(unique)} unique records")

    all_successes = unique

    # Trim to target
    all_successes = all_successes[:target]

    # Assign IDs (final, sequential)
    for i, rec in enumerate(all_successes):
        rec["id"] = f"tr_{i:04d}"

    # Write canonical output (final, dedup'd).
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for rec in all_successes:
            fh.write(json.dumps(rec) + "\n")

    tqdm.write(f"Wrote {len(all_successes)} records to {out_path}")
    return all_successes


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate tool_required.jsonl")
    parser.add_argument("--seeds", default=SEEDS_FILE, help="Path to seeds.jsonl")
    parser.add_argument("--out", default=OUT_FILE, help="Output JSONL path")
    parser.add_argument(
        "--target",
        type=int,
        default=TARGET,
        help="Number of records to generate (default 384; use 4 for smoke test)",
    )
    parser.add_argument(
        "--q-per-seed",
        type=int,
        default=Q_AS_PER_SEED,
        help=(
            f"Max Q&As Opus may produce per seed (default {Q_AS_PER_SEED}). "
            "Higher = more raw records but more pressure on Opus to find distinct angles."
        ),
    )
    args = parser.parse_args()

    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)
    asyncio.run(
        generate(
            args.seeds,
            args.out,
            args.target,
            client,
            q_per_seed=args.q_per_seed,
        )
    )


if __name__ == "__main__":
    main()
