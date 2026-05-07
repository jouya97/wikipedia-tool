"""Tests for eval/stats.py.

No mocks — pure arithmetic verification.
"""

from __future__ import annotations

import pytest

from eval.stats import compute_metrics, _binary_metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rec(
    *,
    tool_required_label: bool,
    tool_called: bool,
    answer_correct: bool,
    n_tool_calls: int = 0,
    question_id: str = "tr_0000",
    tool_description_id: str = "baseline",
    system_prompt_id: str = "baseline",
    run_id: str = "test_run",
) -> dict:
    return {
        "question_id": question_id,
        "tool_required_label": tool_required_label,
        "tool_called": tool_called,
        "answer_correct": answer_correct,
        "n_tool_calls": n_tool_calls,
        "tool_description_id": tool_description_id,
        "system_prompt_id": system_prompt_id,
        "run_id": run_id,
    }


# ---------------------------------------------------------------------------
# _binary_metrics unit tests
# ---------------------------------------------------------------------------

class TestBinaryMetrics:
    def test_perfect_classifier(self):
        m = _binary_metrics(tp=10, fp=0, tn=10, fn=0)
        assert m["accuracy"] == pytest.approx(1.0)
        assert m["precision"] == pytest.approx(1.0)
        assert m["recall"] == pytest.approx(1.0)

    def test_zero_positive_predictions(self):
        """precision should be 0.0 when no positive predictions."""
        m = _binary_metrics(tp=0, fp=0, tn=5, fn=5)
        assert m["precision"] == 0.0
        assert m["recall"] == 0.0  # tp=0, fn=5 → 0/(0+5) = 0

    def test_zero_actual_positives(self):
        """recall should be 0.0 when no actual positives in dataset."""
        m = _binary_metrics(tp=0, fp=3, tn=7, fn=0)
        assert m["recall"] == 0.0  # tp+fn = 0
        assert m["precision"] == 0.0  # tp/(tp+fp) = 0/3 = 0

    def test_all_tp(self):
        m = _binary_metrics(tp=10, fp=0, tn=0, fn=0)
        assert m["accuracy"] == pytest.approx(1.0)
        assert m["precision"] == pytest.approx(1.0)
        assert m["recall"] == pytest.approx(1.0)

    def test_all_tn(self):
        m = _binary_metrics(tp=0, fp=0, tn=10, fn=0)
        assert m["accuracy"] == pytest.approx(1.0)
        assert m["precision"] == 0.0  # no positive predictions
        assert m["recall"] == 0.0     # no actual positives

    def test_zero_n(self):
        m = _binary_metrics(tp=0, fp=0, tn=0, fn=0)
        assert m["accuracy"] == 0.0
        assert m["precision"] == 0.0
        assert m["recall"] == 0.0

    def test_counts_preserved(self):
        m = _binary_metrics(tp=3, fp=2, tn=4, fn=1)
        assert m["tp"] == 3
        assert m["fp"] == 2
        assert m["tn"] == 4
        assert m["fn"] == 1

    def test_partial_precision_recall(self):
        # tp=4, fp=1, tn=3, fn=2 → n=10
        m = _binary_metrics(tp=4, fp=1, tn=3, fn=2)
        assert m["accuracy"] == pytest.approx(7 / 10)
        assert m["precision"] == pytest.approx(4 / 5)
        assert m["recall"] == pytest.approx(4 / 6)


# ---------------------------------------------------------------------------
# compute_metrics tests
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_perfect_tool_use_all_correct(self):
        """All predictions match labels — accuracy, precision, recall = 1.0."""
        records = [
            _rec(tool_required_label=True, tool_called=True, answer_correct=True, question_id=f"tr_{i}", n_tool_calls=1)
            for i in range(4)
        ] + [
            _rec(tool_required_label=False, tool_called=False, answer_correct=True, question_id=f"tnr_{i}", n_tool_calls=0)
            for i in range(4)
        ]
        m = compute_metrics(records)
        assert m["n"] == 8
        assert m["tool_use"]["accuracy"] == pytest.approx(1.0)
        assert m["tool_use"]["precision"] == pytest.approx(1.0)
        assert m["tool_use"]["recall"] == pytest.approx(1.0)
        assert m["tool_use"]["tp"] == 4
        assert m["tool_use"]["tn"] == 4
        assert m["tool_use"]["fp"] == 0
        assert m["tool_use"]["fn"] == 0

    def test_tool_use_tp_tn_fp_fn_counts(self):
        """Manually verify each confusion-matrix cell."""
        records = [
            # TP: required=True, called=True
            _rec(tool_required_label=True, tool_called=True, answer_correct=True, question_id="tp"),
            # TN: required=False, called=False
            _rec(tool_required_label=False, tool_called=False, answer_correct=False, question_id="tn"),
            # FP: required=False, called=True
            _rec(tool_required_label=False, tool_called=True, answer_correct=False, question_id="fp"),
            # FN: required=True, called=False
            _rec(tool_required_label=True, tool_called=False, answer_correct=False, question_id="fn"),
        ]
        m = compute_metrics(records)
        tu = m["tool_use"]
        assert tu["tp"] == 1
        assert tu["tn"] == 1
        assert tu["fp"] == 1
        assert tu["fn"] == 1
        assert tu["accuracy"] == pytest.approx(2 / 4)
        assert tu["precision"] == pytest.approx(1 / 2)  # tp/(tp+fp)
        assert tu["recall"] == pytest.approx(1 / 2)     # tp/(tp+fn)

    def test_n_tool_calls_avg(self):
        records = [
            _rec(tool_required_label=True, tool_called=True, answer_correct=True, n_tool_calls=2, question_id="a"),
            _rec(tool_required_label=True, tool_called=True, answer_correct=True, n_tool_calls=4, question_id="b"),
            _rec(tool_required_label=False, tool_called=False, answer_correct=False, n_tool_calls=0, question_id="c"),
        ]
        m = compute_metrics(records)
        assert m["n_tool_calls_avg"] == pytest.approx((2 + 4 + 0) / 3)

    def test_n_tool_calls_avg_zero(self):
        records = [
            _rec(tool_required_label=False, tool_called=False, answer_correct=False, n_tool_calls=0, question_id=f"x{i}")
            for i in range(5)
        ]
        m = compute_metrics(records)
        assert m["n_tool_calls_avg"] == pytest.approx(0.0)

    def test_answer_correctness_only_tool_required(self):
        """Stable records do NOT contribute to answer_correctness."""
        records = [
            _rec(tool_required_label=True, tool_called=True, answer_correct=True, question_id=f"tr_{i}")
            for i in range(6)
        ] + [
            # Stable records — should be ignored by answer_correctness aggregation.
            _rec(tool_required_label=False, tool_called=False, answer_correct=False, question_id=f"tnr_{i}")
            for i in range(3)
        ]
        m = compute_metrics(records)
        ac = m["answer_correctness"]
        # Only tool_required (n=6) shows up.
        assert ac["n"] == 6
        assert ac["n_correct"] == 6
        assert ac["accuracy"] == pytest.approx(1.0)
        # tool_not_required is its own block (tool-call rate only).
        tnr = m["tool_not_required"]
        assert tnr == {"n": 3, "n_tool_called": 0, "pct_tool_called": pytest.approx(0.0)}

    def test_answer_correctness_with_breakdowns(self):
        records = [
            _rec(tool_required_label=True, tool_called=True, answer_correct=True, question_id="a"),
            _rec(tool_required_label=True, tool_called=False, answer_correct=False, question_id="b"),
            _rec(tool_required_label=True, tool_called=True, answer_correct=True, question_id="c"),
            # Stable: ignored for answer_correctness, but counts in tool_not_required block.
            _rec(tool_required_label=False, tool_called=True, answer_correct=False, question_id="d"),
            _rec(tool_required_label=False, tool_called=False, answer_correct=True, question_id="e"),
        ]
        m = compute_metrics(records)
        ac = m["answer_correctness"]
        assert ac["n"] == 3
        assert ac["n_correct"] == 2
        assert ac["accuracy"] == pytest.approx(2 / 3)
        # Tool-called rollup on tool_required slice.
        assert ac["n_tool_called"] == 2
        assert ac["pct_tool_called"] == pytest.approx(2 / 3)
        # by_tool_called slices.
        assert ac["by_tool_called"]["true"] == {"n": 2, "n_correct": 2, "accuracy": pytest.approx(1.0)}
        assert ac["by_tool_called"]["false"] == {"n": 1, "n_correct": 0, "accuracy": pytest.approx(0.0)}
        # tool_not_required block: 1 of 2 had a tool call.
        assert m["tool_not_required"] == {
            "n": 2, "n_tool_called": 1, "pct_tool_called": pytest.approx(0.5),
        }

    def test_answer_correctness_by_category(self):
        records = [
            _rec(tool_required_label=True, tool_called=True, answer_correct=True, question_id="a"),
            _rec(tool_required_label=True, tool_called=True, answer_correct=False, question_id="b"),
            _rec(tool_required_label=True, tool_called=True, answer_correct=True, question_id="c"),
        ]
        records[0]["category"] = "tv"
        records[1]["category"] = "tv"
        records[2]["category"] = "sports"
        m = compute_metrics(records)
        by_cat = m["answer_correctness"]["by_category"]
        assert by_cat["tv"] == {"n": 2, "n_correct": 1, "accuracy": pytest.approx(0.5)}
        assert by_cat["sports"] == {"n": 1, "n_correct": 1, "accuracy": pytest.approx(1.0)}

    def test_empty_records(self):
        """Empty input should return zero-filled summary without crashing."""
        m = compute_metrics([])
        assert m["n"] == 0
        assert m["n_tool_calls_avg"] == 0.0
        assert m["tool_use"]["accuracy"] == 0.0
        assert m["answer_correctness"]["accuracy"] == 0.0
        assert m["answer_correctness"]["n"] == 0
        assert m["tool_not_required"]["n"] == 0

    def test_output_schema_keys(self):
        """Output dict must have all keys from the run-summary schema."""
        records = [
            _rec(tool_required_label=True, tool_called=True, answer_correct=True, question_id="q1")
        ]
        m = compute_metrics(records)
        for top in ("run_id", "tool_description_id", "system_prompt_id", "n",
                    "tool_use", "answer_correctness", "tool_not_required",
                    "n_tool_calls_avg"):
            assert top in m

        for field in ("accuracy", "precision", "recall", "tp", "fp", "tn", "fn"):
            assert field in m["tool_use"], f"Missing tool_use.{field}"

        ac = m["answer_correctness"]
        for k in ("n", "n_correct", "accuracy", "n_tool_called",
                  "pct_tool_called", "by_tool_called", "by_category"):
            assert k in ac, f"Missing answer_correctness.{k}"

        for k in ("n", "n_tool_called", "pct_tool_called"):
            assert k in m["tool_not_required"], f"Missing tool_not_required.{k}"

    def test_all_false_positive_tool_use(self):
        """All tool calls made but none required — high FP, zero TP."""
        records = [
            _rec(
                tool_required_label=False,
                tool_called=True,
                answer_correct=False,
                question_id=f"fp_{i}",
            )
            for i in range(5)
        ]
        m = compute_metrics(records)
        tu = m["tool_use"]
        assert tu["tp"] == 0
        assert tu["fp"] == 5
        assert tu["tn"] == 0
        assert tu["fn"] == 0
        assert tu["accuracy"] == pytest.approx(0.0)
        assert tu["precision"] == pytest.approx(0.0)
        assert tu["recall"] == 0.0  # no actual positives

    def test_all_false_negative_tool_use(self):
        """Tool never called but always required — high FN, zero TP."""
        records = [
            _rec(
                tool_required_label=True,
                tool_called=False,
                answer_correct=False,
                question_id=f"fn_{i}",
            )
            for i in range(5)
        ]
        m = compute_metrics(records)
        tu = m["tool_use"]
        assert tu["tp"] == 0
        assert tu["fp"] == 0
        assert tu["tn"] == 0
        assert tu["fn"] == 5
        assert tu["accuracy"] == pytest.approx(0.0)
        assert tu["precision"] == 0.0  # no positive predictions
        assert tu["recall"] == pytest.approx(0.0)

    def test_n_tool_calls_none_handled(self):
        """Records with n_tool_calls=None should be treated as 0."""
        records = [
            {
                "question_id": "x",
                "tool_required_label": False,
                "tool_called": False,
                "answer_correct": True,
                "n_tool_calls": None,
                "tool_description_id": "baseline",
                "system_prompt_id": "baseline",
                "run_id": "",
            }
        ]
        m = compute_metrics(records)
        assert m["n_tool_calls_avg"] == pytest.approx(0.0)
