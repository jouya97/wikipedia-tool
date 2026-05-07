"""McNemar's test for comparing two evaluation runs.

Usage (CLI):
    # Pass run directories (preferred — runs/<ts>__td-X__sp-Y/results.jsonl):
    python -m eval.compare runs/20260504-120000__td-baseline__sp-baseline \\
                           runs/20260504-130000__td-v1__sp-v1 \\
                           --metric answer_correct

    # Or pass results.jsonl paths directly:
    python -m eval.compare runs/A/results.jsonl runs/B/results.jsonl

Output (stdout, JSON):
    {
      "run_a": "<path>",
      "run_b": "<path>",
      "metric": "answer_correct",
      "b": <int>,      # run_a=1, run_b=0
      "c": <int>,      # run_a=0, run_b=1
      "p_value": <float>,
      "delta_accuracy": <float>   # accuracy_b - accuracy_a
    }

McNemar's exact test:
    Discordant pairs: b (a correct, b wrong) and c (a wrong, b correct).
    Two-sided exact binomial on min(b, c) successes in b+c trials at p=0.5.
    Uses ``scipy.stats.contingency.mcnemar`` when scipy is available;
    falls back to a hand-rolled exact binomial otherwise.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _resolve_results_path(path: str) -> str:
    """Resolve a CLI argument to a concrete results.jsonl path.

    Accepts either a results JSONL file or a run directory (in which case
    ``<dir>/results.jsonl`` is returned). Raises ``FileNotFoundError`` if
    neither resolves.
    """
    p = Path(path)
    if p.is_dir():
        candidate = p / "results.jsonl"
        if candidate.exists():
            return str(candidate)
        raise FileNotFoundError(
            f"{p} is a directory but {candidate} does not exist"
        )
    if p.exists():
        return str(p)
    raise FileNotFoundError(f"No file or directory at {p}")


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of dicts."""
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _index_by_question_id(
    records: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Return a mapping from question_id to record."""
    idx: dict[str, dict[str, Any]] = {}
    for rec in records:
        qid = rec.get("question_id")
        if qid is not None:
            idx[qid] = rec
    return idx


# ---------------------------------------------------------------------------
# McNemar's exact test (two-sided binomial)
# ---------------------------------------------------------------------------

def _exact_binomial_p(b: int, c: int) -> float:
    """Two-sided exact binomial p-value for McNemar's test.

    Under H0: P(pair discordant in direction A) = 0.5.
    The test statistic is the number of discordant pairs in one direction.
    Two-sided p = 2 * P(X <= min(b,c)) where X ~ Binomial(b+c, 0.5).

    Returns 1.0 when b + c == 0 (no discordant pairs — cannot reject H0).
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # Cumulative binomial P(X <= k) at p=0.5
    # P(X = i) = C(n, i) * 0.5**n
    log_half_n = n * math.log(0.5)
    cumulative = 0.0
    log_comb = 0.0  # log C(n, 0)
    for i in range(k + 1):
        if i > 0:
            log_comb += math.log(n - i + 1) - math.log(i)
        cumulative += math.exp(log_comb + log_half_n)

    p_two_sided = min(1.0, 2.0 * cumulative)
    return p_two_sided


def _mcnemar_p(b: int, c: int) -> float:
    """Compute McNemar's exact two-sided p-value.

    Prefers scipy if available.
    """
    try:
        from scipy.stats.contingency import mcnemar  # type: ignore[import]
        # scipy's mcnemar expects a 2x2 table: [[a, b], [c, d]]
        # where b = (a=1, b=0) and c = (a=0, b=1).
        # We don't have a/d counts here — set them to 0 (they don't affect the test).
        table = [[0, b], [c, 0]]
        result = mcnemar(table, exact=True, correction=False)
        return float(result.pvalue)
    except (ImportError, Exception):
        return _exact_binomial_p(b, c)


# ---------------------------------------------------------------------------
# Public: compare_runs
# ---------------------------------------------------------------------------

def compare_runs(
    run_a_path: str,
    run_b_path: str,
    metric: str = "answer_correct",
) -> dict[str, Any]:
    """Compare two result JSONL files using McNemar's exact test.

    Parameters
    ----------
    run_a_path, run_b_path:
        Either a results JSONL file or a run directory. If a directory is
        given, ``<dir>/results.jsonl`` is used.
    metric:
        The boolean field to compare (default: ``"answer_correct"``).
        Can also be ``"tool_use_correct"`` or any other boolean field.

    Returns
    -------
    dict with keys: run_a, run_b, metric, b, c, p_value, delta_accuracy.
    """
    run_a_path = _resolve_results_path(run_a_path)
    run_b_path = _resolve_results_path(run_b_path)
    records_a = _load_jsonl(run_a_path)
    records_b = _load_jsonl(run_b_path)

    idx_a = _index_by_question_id(records_a)
    idx_b = _index_by_question_id(records_b)

    # Intersect on shared question_ids (in deterministic order).
    shared_ids = sorted(set(idx_a) & set(idx_b))

    b = 0  # run_a correct, run_b wrong
    c = 0  # run_a wrong, run_b correct
    n_a_correct = 0
    n_b_correct = 0
    n_pairs = len(shared_ids)

    for qid in shared_ids:
        val_a = bool(idx_a[qid].get(metric, False))
        val_b = bool(idx_b[qid].get(metric, False))
        if val_a:
            n_a_correct += 1
        if val_b:
            n_b_correct += 1
        if val_a and not val_b:
            b += 1
        elif not val_a and val_b:
            c += 1

    acc_a = n_a_correct / n_pairs if n_pairs > 0 else 0.0
    acc_b = n_b_correct / n_pairs if n_pairs > 0 else 0.0
    delta_accuracy = acc_b - acc_a

    p_value = _mcnemar_p(b, c)

    return {
        "run_a": run_a_path,
        "run_b": run_b_path,
        "metric": metric,
        "b": b,
        "c": c,
        "p_value": p_value,
        "delta_accuracy": delta_accuracy,
    }


# ---------------------------------------------------------------------------
# Paired t-test for numeric metrics (e.g., n_tool_calls)
# ---------------------------------------------------------------------------

def _paired_t_test(diffs: list[float]) -> tuple[float, float, int]:
    """Two-sided paired t-test on a list of (a - b) differences.

    Returns (t_statistic, p_value, df). Uses scipy when available; falls back
    to a hand-rolled t-statistic + survival-function approximation otherwise.

    Edge cases:
        - n < 2 → returns (0.0, 1.0, 0).
        - all diffs identical (sd=0) → returns (inf or 0.0, 1.0 if mean=0 else 0.0, n-1).
    """
    n = len(diffs)
    if n < 2:
        return 0.0, 1.0, 0

    mean_d = sum(diffs) / n
    var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
    sd_d = math.sqrt(var_d)

    if sd_d == 0.0:
        return (0.0, 1.0, n - 1) if mean_d == 0.0 else (math.inf, 0.0, n - 1)

    t_stat = mean_d / (sd_d / math.sqrt(n))
    df = n - 1

    try:
        from scipy import stats as _stats  # type: ignore[import]
        p_two_sided = float(2.0 * _stats.t.sf(abs(t_stat), df))
    except Exception:
        # Wilson-Hilferty-style normal approximation for large df; warn-ish
        # otherwise. Acceptable since scipy is in requirements.txt.
        from math import erfc, sqrt
        p_two_sided = float(erfc(abs(t_stat) / sqrt(2.0)))

    return float(t_stat), float(p_two_sided), df


def compare_tool_calls(
    run_a_path: str,
    run_b_path: str,
    *,
    conditional_on_called: bool = False,
) -> dict[str, Any]:
    """Paired t-test on per-question ``n_tool_calls`` between two runs.

    Parameters
    ----------
    run_a_path, run_b_path:
        Either a results JSONL file or a run directory.
    conditional_on_called:
        If True, restrict the comparison to questions where BOTH runs had
        ``tool_called=True``. Useful for asking "given that both runs decided
        to search, did one iterate more times than the other?"

    Returns
    -------
    dict with keys: run_a, run_b, n_pairs, mean_a, mean_b, delta,
                    t_statistic, p_value, df, conditional_on_called.
    """
    run_a_path = _resolve_results_path(run_a_path)
    run_b_path = _resolve_results_path(run_b_path)

    idx_a = _index_by_question_id(_load_jsonl(run_a_path))
    idx_b = _index_by_question_id(_load_jsonl(run_b_path))
    shared_ids = sorted(set(idx_a) & set(idx_b))

    a_vals: list[float] = []
    b_vals: list[float] = []
    for qid in shared_ids:
        ra, rb = idx_a[qid], idx_b[qid]
        if conditional_on_called and not (
            ra.get("tool_called") and rb.get("tool_called")
        ):
            continue
        try:
            a_vals.append(float(ra.get("n_tool_calls") or 0))
            b_vals.append(float(rb.get("n_tool_calls") or 0))
        except (TypeError, ValueError):
            continue

    n = len(a_vals)
    mean_a = sum(a_vals) / n if n else 0.0
    mean_b = sum(b_vals) / n if n else 0.0
    diffs = [a - b for a, b in zip(a_vals, b_vals)]
    t_stat, p_value, df = _paired_t_test(diffs)

    return {
        "run_a": run_a_path,
        "run_b": run_b_path,
        "n_pairs": n,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "delta": mean_b - mean_a,
        "t_statistic": t_stat,
        "p_value": p_value,
        "df": df,
        "conditional_on_called": conditional_on_called,
    }


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare two evaluation runs (McNemar for binary metrics, paired t-test for n_tool_calls)."
    )
    parser.add_argument(
        "run_a",
        help="Run A: either a results.jsonl path or a run directory",
    )
    parser.add_argument(
        "run_b",
        help="Run B: either a results.jsonl path or a run directory",
    )
    parser.add_argument(
        "--metric",
        default="answer_correct",
        help=(
            "Metric to compare. Boolean fields (answer_correct, "
            "tool_use_correct) use McNemar; 'n_tool_calls' triggers a paired "
            "t-test."
        ),
    )
    parser.add_argument(
        "--conditional-on-called",
        action="store_true",
        help=(
            "For n_tool_calls: restrict to questions where BOTH runs called "
            "the tool. Ignored for binary metrics."
        ),
    )
    args = parser.parse_args()

    if args.metric == "n_tool_calls":
        result = compare_tool_calls(
            args.run_a,
            args.run_b,
            conditional_on_called=args.conditional_on_called,
        )
    else:
        result = compare_runs(args.run_a, args.run_b, metric=args.metric)
    print(json.dumps(result, indent=2))
    sys.exit(0)
