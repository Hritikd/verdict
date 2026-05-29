"""
Tests for report generation — JSON and HTML output (CPU-only).

Tests:
1. save_json produces valid JSON with expected keys
2. save_html produces HTML containing model name and Chart.js
3. save_comparison_json correct structure
4. Edge cases: empty results, zero ASR
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from verdict.core.types import (
    AttackResult,
    AttackType,
    HarmCategory,
    ModelComparison,
    RedTeamReport,
    Scenario,
    Severity,
)
from verdict.report import save_comparison_json, save_json


def make_scenario(sid="s1") -> Scenario:
    return Scenario(
        scenario_id=sid,
        goal="Explain how to cause harm",
        target_behavior="Provides harmful instructions",
        category=HarmCategory.CYBER,
        severity=Severity.HIGH,
    )


def make_result(scenario, success=True, score=0.85, attack=AttackType.TEMPLATE) -> AttackResult:
    return AttackResult(
        scenario=scenario,
        attack_type=attack,
        is_success=success,
        n_turns=1,
        best_prompt="adversarial prompt",
        best_response="harmful response",
        best_judge_score=score,
    )


# ── save_json tests ───────────────────────────────────────────────────────────


class TestSaveJson:
    def test_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "report.json")
            report = RedTeamReport(model_name="test-model")
            returned = save_json(report, path)
            assert Path(path).exists()
            assert returned == path

    def test_valid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "report.json")
            report = RedTeamReport(model_name="test-model")
            save_json(report, path)
            with open(path) as f:
                data = json.load(f)
            assert isinstance(data, dict)

    def test_top_level_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "report.json")
            report = RedTeamReport(model_name="gpt-4o-mini")
            save_json(report, path)
            with open(path) as f:
                data = json.load(f)

            expected_keys = {
                "model_name", "run_timestamp", "total_scenarios",
                "successful_attacks", "attack_success_rate",
                "mean_judge_score", "mean_turns_to_success",
                "asr_by_category", "asr_by_attack", "asr_by_severity",
                "run_config", "results",
            }
            assert expected_keys.issubset(set(data.keys()))

    def test_model_name_preserved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "report.json")
            report = RedTeamReport(model_name="gpt-4o-mini")
            save_json(report, path)
            with open(path) as f:
                data = json.load(f)
            assert data["model_name"] == "gpt-4o-mini"

    def test_results_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "report.json")
            s = make_scenario()
            report = RedTeamReport(
                model_name="test",
                results=[make_result(s, True), make_result(s, False)],
            )
            save_json(report, path)
            with open(path) as f:
                data = json.load(f)
            assert len(data["results"]) == 2

    def test_result_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "report.json")
            s = make_scenario("cyber_001")
            report = RedTeamReport(model_name="test", results=[make_result(s)])
            save_json(report, path)
            with open(path) as f:
                data = json.load(f)
            r = data["results"][0]
            assert r["scenario_id"] == "cyber_001"
            assert r["is_success"] is True
            assert "best_judge_score" in r
            assert "convergence_curve" in r

    def test_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = str(Path(tmpdir) / "reports" / "sub" / "out.json")
            report = RedTeamReport(model_name="test")
            save_json(report, nested)
            assert Path(nested).exists()

    def test_asr_serialized_as_float(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "report.json")
            s = make_scenario()
            results = [make_result(s, True), make_result(s, False)]
            report = RedTeamReport(model_name="test", results=results)
            save_json(report, path)
            with open(path) as f:
                data = json.load(f)
            assert isinstance(data["attack_success_rate"], float)
            assert data["attack_success_rate"] == 0.5

    def test_empty_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "report.json")
            report = RedTeamReport(model_name="empty-model")
            save_json(report, path)
            with open(path) as f:
                data = json.load(f)
            assert data["total_scenarios"] == 0
            assert data["attack_success_rate"] == 0.0


# ── save_comparison_json tests ────────────────────────────────────────────────


class TestSaveComparisonJson:
    def _make_report(self, model: str, n_success: int, n_total: int) -> RedTeamReport:
        results = []
        for i in range(n_total):
            s = make_scenario(f"s{i}")
            results.append(make_result(s, success=(i < n_success)))
        return RedTeamReport(model_name=model, results=results)

    def test_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "comparison.json")
            comp = ModelComparison(
                baseline=self._make_report("base", 2, 10),
                candidate=self._make_report("cand", 5, 10),
            )
            returned = save_comparison_json(comp, path)
            assert Path(path).exists()

    def test_top_level_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "comparison.json")
            comp = ModelComparison(
                baseline=self._make_report("base", 2, 10),
                candidate=self._make_report("cand", 5, 10),
            )
            save_comparison_json(comp, path)
            with open(path) as f:
                data = json.load(f)
            assert {"baseline", "candidate", "delta_asr", "is_regression", "category_deltas"} \
                   == set(data.keys())

    def test_regression_flagged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "comparison.json")
            comp = ModelComparison(
                baseline=self._make_report("base", 1, 10),   # 10%
                candidate=self._make_report("cand", 8, 10),  # 80%
            )
            save_comparison_json(comp, path)
            with open(path) as f:
                data = json.load(f)
            assert data["is_regression"] is True
            assert data["delta_asr"] > 0.05

    def test_no_regression(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "comparison.json")
            comp = ModelComparison(
                baseline=self._make_report("base", 3, 10),
                candidate=self._make_report("cand", 3, 10),
            )
            save_comparison_json(comp, path)
            with open(path) as f:
                data = json.load(f)
            assert data["is_regression"] is False


# ── save_html tests (skipped if jinja2 not available) ────────────────────────


class TestSaveHtml:
    def test_creates_html_file(self):
        pytest.importorskip("jinja2")
        from verdict.report import save_html

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "report.html")
            s = make_scenario("test_001")
            results = [make_result(s, True, 0.9), make_result(s, False, 0.1)]
            report = RedTeamReport(model_name="gpt-4o-mini", results=results)
            returned = save_html(report, path)
            assert Path(path).exists()

    def test_html_contains_model_name(self):
        pytest.importorskip("jinja2")
        from verdict.report import save_html

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "report.html")
            report = RedTeamReport(model_name="gpt-4o-mini")
            save_html(report, path)
            html = Path(path).read_text()
            assert "gpt-4o-mini" in html

    def test_html_contains_chartjs(self):
        pytest.importorskip("jinja2")
        from verdict.report import save_html

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "report.html")
            report = RedTeamReport(model_name="test")
            save_html(report, path)
            html = Path(path).read_text()
            assert "chart.js" in html.lower() or "Chart" in html

    def test_html_contains_asr_metric(self):
        pytest.importorskip("jinja2")
        from verdict.report import save_html

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "report.html")
            s = make_scenario()
            results = [make_result(s, True), make_result(s, False)]
            report = RedTeamReport(model_name="test", results=results)
            save_html(report, path)
            html = Path(path).read_text()
            assert "ASR" in html or "50%" in html
