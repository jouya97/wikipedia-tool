"""Aggregate metrics from a list of graded records.

Two surfaces:

1. **Tool-use** — a real binary classification over ALL records. Positive
   class = ``tool_required_label is True``; predicted positive =
   ``tool_called is True``. Reported as a confusion matrix with
   accuracy/precision/recall. This is the only signal that matters for
   tool_not_required questions: did the agent skip the tool when it should
   have?

2. **Answer correctness** — only meaningful for tool_required records (stable
   Q&As are essentially a free 95-100% from pretrain). Reported as
   ``n / n_correct / accuracy`` plus breakdowns by tool_called and category.

Output schema (``runs/<run_id>/summary.json``)::

    {
      "run_id": ..., "tool_description_id": ..., "system_prompt_id": ...,
      "n": <int>, "n_tool_calls_avg": <float>,
      "tool_use": {accuracy, precision, recall, tp, fp, tn, fn},
      "answer_correctness": {
        "n": <tool_required count>, "n_correct": <int>, "accuracy": <float>,
        "by_tool_called": {"true": {...}, "false": {...}},
        "by_category":    {<cat>: {...}, ...}
      }
    }
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def _binary_metrics(tp: int, fp: int, tn: int, fn: int) -> dict[str, Any]:
    """Confusion-matrix → accuracy/precision/recall (with safe zero-division)."""
    n = tp + fp + tn + fn
    accuracy = (tp + tn) / n if n > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def _accuracy_slice(records: list[dict]) -> dict[str, Any]:
    """{n, n_correct, accuracy} for a subset of graded records."""
    n = len(records)
    n_correct = sum(1 for r in records if bool(r.get("answer_correct", False)))
    return {
        "n": n,
        "n_correct": n_correct,
        "accuracy": (n_correct / n) if n > 0 else 0.0,
    }


def _split_by(records: list[dict], key: str) -> dict[Any, list[dict]]:
    out: dict[Any, list[dict]] = defaultdict(list)
    for r in records:
        out[r.get(key)].append(r)
    return out


def compute_metrics(graded_records: list[dict]) -> dict:
    """Compute aggregate metrics from a list of graded records.

    Each record should have at minimum:
        - ``tool_required_label`` (bool)
        - ``tool_called`` (bool)
        - ``answer_correct`` (bool)
        - ``n_tool_calls`` (int, optional; missing/None treated as 0)
        - ``category`` (str, optional; surfaces in by_category breakdown)
        - ``question_id`` (str)
        - ``tool_description_id`` (str)
        - ``system_prompt_id`` (str)
        - ``run_id`` (str)
    """
    n = len(graded_records)

    # --- Tool-use confusion matrix -------------------------------------------
    tu_tp = tu_fp = tu_tn = tu_fn = 0
    total_tool_calls: float = 0.0

    for rec in graded_records:
        label: bool = bool(rec.get("tool_required_label", False))
        predicted: bool = bool(rec.get("tool_called", False))
        if label and predicted:
            tu_tp += 1
        elif not label and predicted:
            tu_fp += 1
        elif not label and not predicted:
            tu_tn += 1
        else:
            tu_fn += 1

        n_calls = rec.get("n_tool_calls")
        if n_calls is not None:
            try:
                total_tool_calls += float(n_calls)
            except (TypeError, ValueError):
                pass

    n_tool_calls_avg = (total_tool_calls / n) if n > 0 else 0.0

    # --- Splits --------------------------------------------------------------
    required = [r for r in graded_records if bool(r.get("tool_required_label", False))]
    not_required = [r for r in graded_records if not bool(r.get("tool_required_label", False))]

    def _tool_call_rate(records: list[dict]) -> dict[str, Any]:
        n_total = len(records)
        n_called = sum(1 for r in records if bool(r.get("tool_called", False)))
        return {
            "n": n_total,
            "n_tool_called": n_called,
            "pct_tool_called": (n_called / n_total) if n_total else 0.0,
        }

    def _by_tool_called(records: list[dict]) -> dict[str, Any]:
        called = [r for r in records if bool(r.get("tool_called", False))]
        not_called = [r for r in records if not bool(r.get("tool_called", False))]
        return {
            "true": _accuracy_slice(called),
            "false": _accuracy_slice(not_called),
        }

    def _by_category(records: list[dict]) -> dict[str, Any]:
        cats = _split_by(records, "category")
        return {
            str(cat): _accuracy_slice(recs)
            for cat, recs in sorted(
                ((c, rs) for c, rs in cats.items() if c is not None),
                key=lambda kv: str(kv[0]),
            )
        }

    # --- Answer-correctness (tool_required only — stable Q&As are not aggregated
    #     here because their answer accuracy is dominated by pretraining and
    #     trivially high; the only signal worth tracking on stable is whether
    #     a tool was called at all, which lives in the tool_use confusion
    #     matrix and the tool_not_required block below).
    answer_correctness = {
        **_accuracy_slice(required),
        **{
            k: v for k, v in _tool_call_rate(required).items() if k != "n"
        },  # add n_tool_called / pct_tool_called without duplicating n
        "by_tool_called": _by_tool_called(required),
        "by_category": _by_category(required),
    }

    # --- Tool-not-required (only tool-call rate matters here) -----------------
    tool_not_required_block = _tool_call_rate(not_required)

    # Pull metadata from first record (all share these in a single run).
    first = graded_records[0] if graded_records else {}
    run_id = first.get("run_id", "")
    tool_description_id = first.get("tool_description_id", "baseline")
    system_prompt_id = first.get("system_prompt_id", "baseline")

    return {
        "run_id": run_id,
        "tool_description_id": tool_description_id,
        "system_prompt_id": system_prompt_id,
        "n": n,
        "n_tool_calls_avg": n_tool_calls_avg,
        "tool_use": _binary_metrics(tu_tp, tu_fp, tu_tn, tu_fn),
        "answer_correctness": answer_correctness,
        "tool_not_required": tool_not_required_block,
    }
