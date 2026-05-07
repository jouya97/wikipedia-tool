"""Tests for eval/compare.py.

Uses temporary JSONL files to avoid any filesystem side-effects.
All McNemar's arithmetic is verified without network calls.
"""

from __future__ import annotations

import json
import math
import os
import tempfile

import pytest

from eval.compare import (
    compare_runs,
    compare_tool_calls,
    _exact_binomial_p,
    _mcnemar_p,
    _paired_t_test,
    _load_jsonl,
    _index_by_question_id,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_jsonl(path: str, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def _make_record(
    question_id: str,
    answer_correct: bool,
    tool_use_correct: bool = True,
) -> dict:
    return {
        "question_id": question_id,
        "answer_correct": answer_correct,
        "tool_use_correct": tool_use_correct,
        "tool_required_label": True,
        "tool_called": True,
        "final_answer": "some answer",
        "golden_answer": "the answer",
        "user_question": "a question",
        "n_tool_calls": 1,
    }


# ---------------------------------------------------------------------------
# _exact_binomial_p unit tests
# ---------------------------------------------------------------------------

class TestExactBinomialP:
    def test_zero_discordant_pairs(self):
        """No discordant pairs → p=1.0 (cannot reject null)."""
        p = _exact_binomial_p(0, 0)
        assert p == pytest.approx(1.0)

    def test_symmetric_discordant_pairs(self):
        """Equal b and c → high p-value (symmetric → cannot reject)."""
        p = _exact_binomial_p(5, 5)
        assert p > 0.5

    def test_highly_asymmetric_small_p(self):
        """Large discordance (b=0, c=20) → very small p-value."""
        p = _exact_binomial_p(0, 20)
        assert p < 0.001

    def test_single_discordant_pair(self):
        """b=1, c=0 (or b=0, c=1) → p=1.0 (single pair can't reject)."""
        p = _exact_binomial_p(1, 0)
        assert p == pytest.approx(1.0)
        p2 = _exact_binomial_p(0, 1)
        assert p2 == pytest.approx(1.0)

    def test_p_value_is_two_sided(self):
        """b=1, c=9 vs b=9, c=1 should give the same p-value."""
        p1 = _exact_binomial_p(1, 9)
        p2 = _exact_binomial_p(9, 1)
        assert p1 == pytest.approx(p2)

    def test_p_value_bounded(self):
        """p-value must always be in [0, 1]."""
        for b, c in [(0, 0), (3, 3), (0, 10), (10, 0), (100, 1), (1, 100)]:
            p = _exact_binomial_p(b, c)
            assert 0.0 <= p <= 1.0, f"p={p} out of bounds for b={b}, c={c}"

    def test_known_value_b2_c8(self):
        """
        b=2, c=8, n=10: X ~ Binomial(10, 0.5), P(X <= 2) = C(10,0)+C(10,1)+C(10,2) / 2^10
        = (1 + 10 + 45) / 1024 = 56/1024 ≈ 0.0547; two-sided ≈ 0.1094
        """
        p = _exact_binomial_p(2, 8)
        expected = 2 * (1 + 10 + 45) / (2 ** 10)
        assert p == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# _mcnemar_p — tests that it runs without error (delegates to scipy or fallback)
# ---------------------------------------------------------------------------

class TestMcNemarP:
    def test_runs_without_error(self):
        p = _mcnemar_p(5, 5)
        assert 0.0 <= p <= 1.0

    def test_large_discordance_small_p(self):
        p = _mcnemar_p(0, 30)
        assert p < 0.001

    def test_no_discordance_p_one(self):
        p = _mcnemar_p(0, 0)
        assert p == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compare_runs integration tests using temp files
# ---------------------------------------------------------------------------

class TestCompareRuns:
    def _runs_with_discordance(self, b: int, c: int, n_concordant: int = 10):
        """
        Build two run dicts pairing on question_id.

        b: number of pairs where a=1, b=0
        c: number of pairs where a=0, b=1
        n_concordant: number of pairs where a==b (split evenly TP and TN)
        """
        records_a = []
        records_b = []
        idx = 0

        # concordant correct (both correct)
        for i in range(n_concordant // 2):
            qid = f"q_{idx:04d}"
            idx += 1
            records_a.append(_make_record(qid, answer_correct=True))
            records_b.append(_make_record(qid, answer_correct=True))

        # concordant wrong (both wrong)
        for i in range(n_concordant - n_concordant // 2):
            qid = f"q_{idx:04d}"
            idx += 1
            records_a.append(_make_record(qid, answer_correct=False))
            records_b.append(_make_record(qid, answer_correct=False))

        # b discordant: a=1, b=0
        for i in range(b):
            qid = f"q_{idx:04d}"
            idx += 1
            records_a.append(_make_record(qid, answer_correct=True))
            records_b.append(_make_record(qid, answer_correct=False))

        # c discordant: a=0, b=1
        for i in range(c):
            qid = f"q_{idx:04d}"
            idx += 1
            records_a.append(_make_record(qid, answer_correct=False))
            records_b.append(_make_record(qid, answer_correct=True))

        return records_a, records_b

    def test_b_c_counts_symmetric(self):
        """Verify that b and c are counted correctly."""
        records_a, records_b = self._runs_with_discordance(b=3, c=7)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as fa, \
             tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as fb:
            _write_jsonl(fa.name, records_a)
            _write_jsonl(fb.name, records_b)
            path_a, path_b = fa.name, fb.name

        try:
            result = compare_runs(path_a, path_b, metric="answer_correct")
            assert result["b"] == 3
            assert result["c"] == 7
        finally:
            os.unlink(path_a)
            os.unlink(path_b)

    def test_no_discordance_high_p(self):
        """Zero discordant pairs → p=1.0."""
        records_a, records_b = self._runs_with_discordance(b=0, c=0, n_concordant=20)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as fa, \
             tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as fb:
            _write_jsonl(fa.name, records_a)
            _write_jsonl(fb.name, records_b)
            path_a, path_b = fa.name, fb.name

        try:
            result = compare_runs(path_a, path_b)
            assert result["p_value"] == pytest.approx(1.0)
        finally:
            os.unlink(path_a)
            os.unlink(path_b)

    def test_high_discordance_low_p(self):
        """Large one-sided discordance → small p-value."""
        records_a, records_b = self._runs_with_discordance(b=0, c=25, n_concordant=10)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as fa, \
             tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as fb:
            _write_jsonl(fa.name, records_a)
            _write_jsonl(fb.name, records_b)
            path_a, path_b = fa.name, fb.name

        try:
            result = compare_runs(path_a, path_b)
            assert result["p_value"] < 0.001
        finally:
            os.unlink(path_a)
            os.unlink(path_b)

    def test_delta_accuracy_positive_when_b_is_superior(self):
        """delta = acc_b - acc_a; if b outperforms a, delta > 0."""
        # b=0 discordant a-wins, c=10 discordant b-wins → b is better
        n_concordant = 10  # 5 both-correct + 5 both-wrong
        records_a, records_b = self._runs_with_discordance(b=0, c=10, n_concordant=n_concordant)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as fa, \
             tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as fb:
            _write_jsonl(fa.name, records_a)
            _write_jsonl(fb.name, records_b)
            path_a, path_b = fa.name, fb.name

        try:
            result = compare_runs(path_a, path_b)
            assert result["delta_accuracy"] > 0
        finally:
            os.unlink(path_a)
            os.unlink(path_b)

    def test_delta_accuracy_negative_when_a_is_superior(self):
        """If a outperforms b, delta_accuracy < 0."""
        records_a, records_b = self._runs_with_discordance(b=10, c=0, n_concordant=10)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as fa, \
             tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as fb:
            _write_jsonl(fa.name, records_a)
            _write_jsonl(fb.name, records_b)
            path_a, path_b = fa.name, fb.name

        try:
            result = compare_runs(path_a, path_b)
            assert result["delta_accuracy"] < 0
        finally:
            os.unlink(path_a)
            os.unlink(path_b)

    def test_delta_accuracy_zero_symmetric(self):
        """Equal b and c → delta_accuracy = 0."""
        records_a, records_b = self._runs_with_discordance(b=5, c=5, n_concordant=10)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as fa, \
             tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as fb:
            _write_jsonl(fa.name, records_a)
            _write_jsonl(fb.name, records_b)
            path_a, path_b = fa.name, fb.name

        try:
            result = compare_runs(path_a, path_b)
            assert result["delta_accuracy"] == pytest.approx(0.0)
        finally:
            os.unlink(path_a)
            os.unlink(path_b)

    def test_output_keys(self):
        """Output dict must have all required keys."""
        records_a, records_b = self._runs_with_discordance(b=2, c=3)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as fa, \
             tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as fb:
            _write_jsonl(fa.name, records_a)
            _write_jsonl(fb.name, records_b)
            path_a, path_b = fa.name, fb.name

        try:
            result = compare_runs(path_a, path_b)
            for key in ("run_a", "run_b", "metric", "b", "c", "p_value", "delta_accuracy"):
                assert key in result, f"Missing key: {key}"
        finally:
            os.unlink(path_a)
            os.unlink(path_b)

    def test_metric_parameter(self):
        """compare_runs should respect the metric parameter."""
        # Build records where tool_use_correct differs between runs
        records_a = [
            _make_record("q1", answer_correct=True, tool_use_correct=True),
            _make_record("q2", answer_correct=True, tool_use_correct=False),
        ]
        records_b = [
            _make_record("q1", answer_correct=True, tool_use_correct=False),
            _make_record("q2", answer_correct=True, tool_use_correct=True),
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as fa, \
             tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as fb:
            _write_jsonl(fa.name, records_a)
            _write_jsonl(fb.name, records_b)
            path_a, path_b = fa.name, fb.name

        try:
            result = compare_runs(path_a, path_b, metric="tool_use_correct")
            assert result["metric"] == "tool_use_correct"
            # b=1 (a=T,b=F), c=1 (a=F,b=T)
            assert result["b"] == 1
            assert result["c"] == 1
        finally:
            os.unlink(path_a)
            os.unlink(path_b)

    def test_unmatched_question_ids_excluded(self):
        """Records with no paired question_id in the other run are ignored."""
        records_a = [
            _make_record("shared_q", answer_correct=True),
            _make_record("only_in_a", answer_correct=True),
        ]
        records_b = [
            _make_record("shared_q", answer_correct=False),
            _make_record("only_in_b", answer_correct=True),
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as fa, \
             tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as fb:
            _write_jsonl(fa.name, records_a)
            _write_jsonl(fb.name, records_b)
            path_a, path_b = fa.name, fb.name

        try:
            # Only "shared_q" is paired: a=True, b=False → b=1, c=0
            result = compare_runs(path_a, path_b)
            assert result["b"] == 1
            assert result["c"] == 0
        finally:
            os.unlink(path_a)
            os.unlink(path_b)

    def test_run_paths_in_output(self):
        """run_a and run_b in the output must match the input paths."""
        records_a, records_b = self._runs_with_discordance(b=1, c=1)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as fa, \
             tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as fb:
            _write_jsonl(fa.name, records_a)
            _write_jsonl(fb.name, records_b)
            path_a, path_b = fa.name, fb.name

        try:
            result = compare_runs(path_a, path_b)
            assert result["run_a"] == path_a
            assert result["run_b"] == path_b
        finally:
            os.unlink(path_a)
            os.unlink(path_b)

    def test_delta_accuracy_value(self):
        """Manually verify delta_accuracy = acc_b - acc_a."""
        # 10 concordant correct, 0 concordant wrong
        # b=2: a correct, b wrong → a gets 2 extra correct
        # c=4: a wrong, b correct → b gets 4 extra correct
        # n = 10 + 0 + 2 + 4 = 16
        # n_a_correct = 10 + 2 = 12 → acc_a = 12/16 = 0.75
        # n_b_correct = 10 + 4 = 14 → acc_b = 14/16 = 0.875
        # delta = 0.875 - 0.75 = 0.125
        records_a, records_b = self._runs_with_discordance(b=2, c=4, n_concordant=10)
        # n_concordant=10: 5 both-correct + 5 both-wrong
        # n_a_correct = 5 + 2 = 7; n_b_correct = 5 + 4 = 9
        # delta = 9/16 - 7/16 = 2/16 = 0.125
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as fa, \
             tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as fb:
            _write_jsonl(fa.name, records_a)
            _write_jsonl(fb.name, records_b)
            path_a, path_b = fa.name, fb.name

        try:
            result = compare_runs(path_a, path_b)
            assert result["delta_accuracy"] == pytest.approx(2 / 16, rel=1e-6)
        finally:
            os.unlink(path_a)
            os.unlink(path_b)


# ---------------------------------------------------------------------------
# Paired t-test on n_tool_calls
# ---------------------------------------------------------------------------

class TestPairedTTest:
    def test_zero_diff_returns_p_one(self):
        t, p, df = _paired_t_test([0.0, 0.0, 0.0, 0.0])
        assert t == 0.0
        assert p == pytest.approx(1.0)
        assert df == 3

    def test_n_lt_2_returns_neutral(self):
        t, p, df = _paired_t_test([1.5])
        assert (t, p, df) == (0.0, 1.0, 0)

    def test_strong_positive_diff_low_p(self):
        # Big consistent difference → small p.
        diffs = [1.0] * 30
        t, p, df = _paired_t_test(diffs)
        # sd=0 here → t=inf, p=0 by our edge-case rule
        assert df == 29
        assert p == pytest.approx(0.0)

    def test_noisy_zero_mean_high_p(self):
        # Symmetric around 0 → p should be near 1.
        diffs = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
        t, p, df = _paired_t_test(diffs)
        assert abs(t) < 1e-10
        assert p == pytest.approx(1.0)


class TestCompareToolCalls:
    def test_basic_paired_difference(self, tmp_path):
        a_path = tmp_path / "a.jsonl"
        b_path = tmp_path / "b.jsonl"
        _write_jsonl(
            str(a_path),
            [
                {"question_id": "q1", "n_tool_calls": 2, "tool_called": True},
                {"question_id": "q2", "n_tool_calls": 3, "tool_called": True},
                {"question_id": "q3", "n_tool_calls": 1, "tool_called": True},
            ],
        )
        _write_jsonl(
            str(b_path),
            [
                {"question_id": "q1", "n_tool_calls": 1, "tool_called": True},
                {"question_id": "q2", "n_tool_calls": 1, "tool_called": True},
                {"question_id": "q3", "n_tool_calls": 0, "tool_called": False},
            ],
        )
        r = compare_tool_calls(str(a_path), str(b_path))
        assert r["n_pairs"] == 3
        assert r["mean_a"] == pytest.approx(2.0)
        assert r["mean_b"] == pytest.approx(2 / 3)
        assert r["delta"] == pytest.approx(2 / 3 - 2.0)
        assert r["t_statistic"] > 0  # diffs = a - b are positive (a > b)
        assert 0.0 < r["p_value"] <= 1.0

    def test_conditional_on_called_filters(self, tmp_path):
        a_path = tmp_path / "a.jsonl"
        b_path = tmp_path / "b.jsonl"
        _write_jsonl(
            str(a_path),
            [
                {"question_id": "q1", "n_tool_calls": 2, "tool_called": True},
                {"question_id": "q2", "n_tool_calls": 5, "tool_called": True},
                {"question_id": "q3", "n_tool_calls": 0, "tool_called": False},
            ],
        )
        _write_jsonl(
            str(b_path),
            [
                {"question_id": "q1", "n_tool_calls": 1, "tool_called": True},
                {"question_id": "q2", "n_tool_calls": 0, "tool_called": False},
                {"question_id": "q3", "n_tool_calls": 0, "tool_called": False},
            ],
        )
        r = compare_tool_calls(
            str(a_path), str(b_path), conditional_on_called=True
        )
        # Only q1 had tool_called=True in both runs.
        assert r["n_pairs"] == 1
        assert r["conditional_on_called"] is True

    def test_directory_input_resolves_results_jsonl(self, tmp_path):
        for name, recs in (
            ("run_a", [{"question_id": "q1", "n_tool_calls": 2, "tool_called": True}]),
            ("run_b", [{"question_id": "q1", "n_tool_calls": 1, "tool_called": True}]),
        ):
            d = tmp_path / name
            d.mkdir()
            _write_jsonl(str(d / "results.jsonl"), recs)
        r = compare_tool_calls(str(tmp_path / "run_a"), str(tmp_path / "run_b"))
        assert r["n_pairs"] == 1
