"""
datagen/gen_stable.py

Generate tool_not_required.jsonl: 68 stable evergreen Q&A pairs that do NOT
require a Wikipedia search — facts that were true before March 2025 and remain
stable.

CLI:
    python -m datagen.gen_stable --out data/tool_not_required.jsonl --target 68

Smoke test:
    python -m datagen.gen_stable --out data/tool_not_required_smoke.jsonl --target 4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
# Quiet by default: only warnings/errors. Set DATAGEN_LOG_LEVEL=INFO for verbose.
_log_level = os.environ.get("DATAGEN_LOG_LEVEL", "WARNING").upper()
logging.basicConfig(level=_log_level, format="%(levelname)s %(message)s")
for _noisy in ("httpx", "httpcore", "anthropic", "urllib3"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL = "claude-opus-4-7"
MAX_TOKENS = 8192
OUT_FILE = "data/tool_not_required.jsonl"
TARGET = 68
BATCH_SIZE = 80  # Ask for 80, trim to 68 after dedup

# ---------------------------------------------------------------------------
# 5-gram Jaccard near-duplicate detection (same logic as gen_tool_required)
# ---------------------------------------------------------------------------

def _ngrams(text: str, n: int = 5) -> set[str]:
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


def dedup_records(records: list[dict], threshold: float = 0.5) -> list[dict]:
    kept: list[dict] = []
    for rec in records:
        q = rec.get("user_question", "")
        is_dup = any(
            _jaccard(q, k.get("user_question", "")) > threshold
            for k in kept
        )
        if not is_dup:
            kept.append(rec)
    return kept


# ---------------------------------------------------------------------------
# Opus prompt for stable data generation
# ---------------------------------------------------------------------------

STABLE_SYSTEM_PROMPT = """\
You are a high-quality synthetic-data generator for an AI evaluation benchmark.
Your task is to produce stable, evergreen Q&A pairs for which NO external lookup
is needed — the answers are well-known, stable facts that any well-educated person
or language model should know without needing to search the web.

OUTPUT FORMAT:
Return a JSON array of exactly {batch_size} objects. Each object MUST follow this
schema exactly:
{{
  "user_question": "<colloquial question a curious person might ask>",
  "golden_answer": "<concise, accurate answer in 1-3 sentences>"
}}

Output ONLY the JSON array — no markdown fences, no commentary, no extra keys.

CONTENT RULES:
1. NO politics, politicians, elections, government policy, or anything that could
   be politically contentious.
2. NO events, developments, or changes that occurred after March 2025. All facts
   must have been stably true well before that date.
3. NO obscure trivia that only specialists would know. Aim for facts a reasonably
   educated person (or a solid language model) would have in training data.
4. Broad subject mix encouraged: science, history, geography, literature, music,
   sports records (pre-March-2025), film, language, food, nature, mathematics,
   technology concepts, etc.
5. Phrase user_question colloquially — like an actual curious person typing,
   NOT like a search query. Use varied registers: casual, context-setting,
   comparative, follow-up-shaped. The question must end with "?".
6. The golden_answer must be factually accurate and self-contained (doesn't
   assume context from a conversation).
7. Ensure diversity — don't cluster too many questions around one domain.
"""


def _build_prompt(batch_size: int) -> str:
    return (
        f"Generate exactly {batch_size} stable, evergreen Q&A pairs following all "
        "the rules in your system prompt. Return ONLY the JSON array."
    )


# ---------------------------------------------------------------------------
# Core generation function
# ---------------------------------------------------------------------------

def generate(
    out_path: str,
    target: int,
    client: object,
    batch_size: int = BATCH_SIZE,
) -> list[dict]:
    """
    Single Opus 4.7 (low effort) call to generate stable Q&A pairs.
    Returns the trimmed list of records.
    """
    system = STABLE_SYSTEM_PROMPT.format(batch_size=batch_size)
    prompt = _build_prompt(batch_size)

    print(f"Calling Opus 4.7 for {batch_size} stable Q&A pairs...", flush=True)

    response = client.messages.create(  # type: ignore[union-attr]
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": prompt}],
        output_config={"effort": "low"},
    )

    raw_text = "".join(
        block.text for block in response.content if block.type == "text"
    ).strip()

    records = _parse_records(raw_text)
    valid = [r for r in records if _validate_record(r)]
    unique = dedup_records(valid)
    print(
        f"Parsed {len(records)} → {len(valid)} valid → {len(unique)} unique",
        flush=True,
    )

    # Trim to target
    trimmed = unique[:target]

    # Assign IDs
    for i, rec in enumerate(trimmed):
        rec["id"] = f"tnr_{i:04d}"

    # Write
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for rec in trimmed:
            fh.write(json.dumps(rec) + "\n")

    print(f"Wrote {len(trimmed)} records to {out_path}", flush=True)
    return trimmed


def _parse_records(text: str) -> list[dict]:
    """Extract a JSON array from the model's response text."""
    # Strip markdown fences
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()

    # Try direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
    except json.JSONDecodeError:
        pass

    # Try to find a JSON array in the text
    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        try:
            obj = json.loads(match.group())
            if isinstance(obj, list):
                return obj
        except json.JSONDecodeError:
            pass

    logger.warning("Could not parse JSON array from response")
    return []


def _validate_record(rec: object) -> bool:
    if not isinstance(rec, dict):
        return False
    q = rec.get("user_question", "")
    a = rec.get("golden_answer", "")
    return bool(q) and bool(a) and q.strip().endswith("?")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate tool_not_required.jsonl")
    parser.add_argument("--out", default=OUT_FILE, help="Output JSONL path")
    parser.add_argument(
        "--target",
        type=int,
        default=TARGET,
        help="Number of Q&A pairs to output (default 68; use 4 for smoke test)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="How many pairs to ask Opus for (trimmed to --target after dedup)",
    )
    args = parser.parse_args()

    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)
    generate(args.out, args.target, client, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
