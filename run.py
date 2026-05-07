"""Run Haiku-agent inference + eval over the tool-required + tool-not-required sets."""
import argparse
import asyncio
import inspect
import json
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from tqdm.asyncio import tqdm as tqdm_asyncio

from agent.haiku_agent import run_agent
from eval.grade import grade_batch
from eval.stats import compute_metrics


REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
RUNS_DIR = REPO_ROOT / "runs"


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run datagen first.")
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_seed_categories() -> dict[str, str]:
    """Map seed title → category from seeds.jsonl. Used to slice eval results by tv/sports."""
    seeds_path = DATA_DIR / "seeds.jsonl"
    if not seeds_path.exists():
        return {}
    cats: dict[str, str] = {}
    for row in _load_jsonl(seeds_path):
        title = row.get("title")
        cat = row.get("category")
        if title and cat:
            cats[title] = cat
    return cats


def load_dataset() -> list[dict]:
    seed_cats = _load_seed_categories()
    examples: list[dict] = []
    for row in _load_jsonl(DATA_DIR / "tool_required.jsonl"):
        row["tool_required_label"] = True
        row["category"] = seed_cats.get(row.get("seed_title"), "unknown")
        examples.append(row)
    for row in _load_jsonl(DATA_DIR / "tool_not_required.jsonl"):
        row["tool_required_label"] = False
        row["category"] = "stable"
        examples.append(row)
    return examples


async def _call_run_agent(question, tool_description_id, system_prompt_id):
    # Tolerates either sync or async run_agent — subagent C may ship either.
    if inspect.iscoroutinefunction(run_agent):
        return await run_agent(
            question, tool_description_id, system_prompt_id=system_prompt_id
        )
    return await asyncio.to_thread(
        run_agent, question, tool_description_id, system_prompt_id=system_prompt_id
    )


async def run_inference(
    examples: list[dict],
    tool_description_id: str,
    system_prompt_id: str,
    concurrency: int,
    output_path: Path | None = None,
) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)

    async def one(example: dict) -> dict:
        async with sem:
            trace = await _call_run_agent(
                example["user_question"], tool_description_id, system_prompt_id
            )
            trace["question_id"] = example["id"]
            trace["tool_required_label"] = example["tool_required_label"]
            trace["category"] = example.get("category")
            trace["user_question"] = example["user_question"]
            trace["golden_answer"] = example["golden_answer"]
            return trace

    tasks = [one(ex) for ex in examples]

    if output_path is None:
        return list(await asyncio.gather(*tasks))

    # Streaming path: tqdm bar + append-as-you-go.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    traces: list[dict] = []
    with open(output_path, "a", encoding="utf-8") as fh:
        for coro in tqdm_asyncio.as_completed(
            tasks,
            total=len(tasks),
            desc="Inference",
            unit="q",
        ):
            trace = await coro
            traces.append(trace)
            fh.write(json.dumps(trace, ensure_ascii=False) + "\n")
            fh.flush()
    return traces


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


async def main(args: argparse.Namespace) -> None:
    load_dotenv()

    examples = load_dataset()
    if args.limit:
        examples = examples[: args.limit]

    n_req = sum(e["tool_required_label"] for e in examples)
    n_not = len(examples) - n_req
    print(
        f"Loaded {len(examples)} examples "
        f"({n_req} tool-required, {n_not} tool-not-required)"
    )

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    # Encode the prompt versions into the dir name so reviewers can tell runs
    # apart at a glance and `eval/compare.py runs/A runs/B` is self-documenting.
    run_dir_name = f"{run_id}__td-{args.tool_description_id}__sp-{args.system_prompt_id}"
    run_dir = RUNS_DIR / run_dir_name
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
        examples,
        args.tool_description_id,
        args.system_prompt_id,
        args.concurrency,
        output_path=inference_path,
    )
    print(f"Inference: {len(traces)} traces in {time.time() - t0:.1f}s — streamed to {inference_path}")

    t0 = time.time()
    graded = await grade_batch(traces, output_path=results_path)
    print(f"Grading: {len(graded)} graded in {time.time() - t0:.1f}s — streamed to {results_path}")

    summary = compute_metrics(graded)
    summary["run_id"] = run_id
    summary["tool_description_id"] = args.tool_description_id
    summary["system_prompt_id"] = args.system_prompt_id
    write_json(summary_path, summary)
    print(f"Summary → {summary_path}")
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Haiku agent inference + eval")
    p.add_argument("--tool-description-id", default="baseline")
    p.add_argument("--system-prompt-id", default="baseline")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--limit", type=int, default=None, help="Cap examples (for smoke runs)")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
