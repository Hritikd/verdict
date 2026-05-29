"""
Tests for core data types — RedTeamReport, AttackResult, ModelComparison.

All tests are CPU-only: no LLM calls. We construct synthetic AttackResult
objects and verify that the aggregate metrics are computed correctly.
"""

from __future__ import annotations

import pytest

from verdict.core.types import (
    AttackResult,
    AttackTurn,
    AttackType,
    HarmCategory,
    JudgeVerdict,
    ModelComparison,
    RedTeamReport,
    Scenario,
    Severity,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


def make_scenario(sid="s1", cat=HarmCategory.CYBER, sev=Severity.HIGH) -> Scenario:
    return Scenario(
        scenario_id=sid,
        goal="Do something harmful",
        target_behavior="Harmful response produced",
        category=cat,
        severity=sev,
    )


def make_turn(n: int, score: float, success: bool) -> AttackTurn:
    return AttackTurn(
        turn_number=n,
        attack_prompt=f"prompt_{n}",
        target_response=f"response_{n}",
        judge_score=score,
        judge_verdict=JudgeVerdict.HARMFUL if success else JudgeVerdict.BENIGN,
        is_success=success,
    )


def make_result(
    scenario: Scenario,
    attack_type: AttackType = AttackType.TEMPLATE,
    is_success: bool = True,
    best_judge_score: float = 0.8,
    n_turns: int = 1,
    turns: list | None = None,
) -> AttackResult:
    return AttackResult(
        scenario=scenario,
        attack_type=attack_type,
        is_success=is_success,
        n_turns=n_turns,
        best_prompt="adversarial prompt",
        best_response="harmful response",
        best_judge_score=best_judge_score,
        turns=turns or [],
    )


# ── Scenario tests ────────────────────────────────────────────────────────────


class TestScenario:
    def test_str_repr(self):
        s = make_scenario("cyber_001")
        assert "cyber_001" in str(s)
        assert "high" in str(s)

    def test_default_category(self):
        s = Scenario(
            scenario_id="x",
            goal="...",
            target_behavior="...",
        )
        assert s.category == HarmCategory.GENERAL
        assert s.severity == Severity.MEDIUM


# ── AttackResult tests ────────────────────────────────────────────────────────


class TestAttackResult:
    def test_convergence_curve_empty(self):
        r = make_result(make_scenario())
        assert r.convergence_curve == []

    def test_convergence_curve_from_turns(self):
        turns = [
            make_turn(1, 0.2, False),
            make_turn(2, 0.5, False),
            make_turn(3, 0.9, True),
        ]
        r = make_result(make_scenario(), turns=turns, n_turns=3)
        assert r.convergence_curve == [0.2, 0.5, 0.9]

    def test_turns_to_success(self):
        turns = [
            make_turn(1, 0.2, False),
            make_turn(2, 0.9, True),
            make_turn(3, 0.95, True),
        ]
        r = make_result(make_scenario(), turns=turns)
        assert r.turns_to_success == 2

    def test_turns_to_success_none(self):
        turns = [make_turn(1, 0.1, False), make_turn(2, 0.2, False)]
        r = make_result(make_scenario(), is_success=False, turns=turns)
        assert r.turns_to_success is None


# ── RedTeamReport tests ───────────────────────────────────────────────────────


class TestRedTeamReport:
    def test_empty_report(self):
        report = RedTeamReport(model_name="test")
        assert report.total_scenarios == 0
        assert report.successful_attacks == 0
        assert report.attack_success_rate == 0.0
        assert report.mean_judge_score == 0.0
        assert report.mean_turns_to_success == 0.0

    def test_asr_all_success(self):
        scenarios = [make_scenario(f"s{i}") for i in range(5)]
        results = [make_result(s, is_success=True, best_judge_score=0.9) for s in scenarios]
        report = RedTeamReport(model_name="test", results=results)
        assert report.attack_success_rate == 1.0
        assert report.successful_attacks == 5

    def test_asr_no_success(self):
        scenarios = [make_scenario(f"s{i}") for i in range(4)]
        results = [make_result(s, is_success=False, best_judge_score=0.1) for s in scenarios]
        report = RedTeamReport(model_name="test", results=results)
        assert report.attack_success_rate == 0.0

    def test_asr_partial(self):
        s1, s2, s3, s4 = [make_scenario(f"s{i}") for i in range(4)]
        results = [
            make_result(s1, is_success=True),
            make_result(s2, is_success=True),
            make_result(s3, is_success=False),
            make_result(s4, is_success=False),
        ]
        report = RedTeamReport(model_name="test", results=results)
        assert report.attack_success_rate == 0.5

    def test_mean_judge_score(self):
        scenarios = [make_scenario(f"s{i}") for i in range(4)]
        scores = [0.2, 0.4, 0.6, 0.8]
        results = [make_result(s, best_judge_score=sc) for s, sc in zip(scenarios, scores)]
        report = RedTeamReport(model_name="test", results=results)
        assert abs(report.mean_judge_score - 0.5) < 1e-9

    def test_asr_by_category(self):
        cyber_s = make_scenario("c1", cat=HarmCategory.CYBER)
        chem_s = make_scenario("ch1", cat=HarmCategory.CHEMICAL)
        results = [
            make_result(cyber_s, is_success=True),
            make_result(chem_s, is_success=False),
        ]
        report = RedTeamReport(model_name="test", results=results)
        asr = report.asr_by_category()
        assert asr[HarmCategory.CYBER.value] == 1.0
        assert asr[HarmCategory.CHEMICAL.value] == 0.0

    def test_asr_by_attack(self):
        s1, s2, s3 = [make_scenario(f"s{i}") for i in range(3)]
        results = [
            make_result(s1, attack_type=AttackType.TEMPLATE, is_success=True),
            make_result(s2, attack_type=AttackType.INJECTION, is_success=True),
            make_result(s3, attack_type=AttackType.INJECTION, is_success=False),
        ]
        report = RedTeamReport(model_name="test", results=results)
        asr = report.asr_by_attack()
        assert asr[AttackType.TEMPLATE.value] == 1.0
        assert abs(asr[AttackType.INJECTION.value] - 0.5) < 1e-9

    def test_asr_by_severity(self):
        critical_s = make_scenario("cr1", sev=Severity.CRITICAL)
        medium_s = make_scenario("m1", sev=Severity.MEDIUM)
        results = [
            make_result(critical_s, is_success=True),
            make_result(medium_s, is_success=False),
        ]
        report = RedTeamReport(model_name="test", results=results)
        asr = report.asr_by_severity()
        assert asr[Severity.CRITICAL.value] == 1.0
        assert asr[Severity.MEDIUM.value] == 0.0

    def test_summary_table_contains_model_name(self):
        report = RedTeamReport(model_name="gpt-4o-mini")
        summary = report.summary_table()
        assert "gpt-4o-mini" in summary

    def test_mean_turns_to_success(self):
        """Mean turns should only count successful attacks."""
        s1, s2, s3 = [make_scenario(f"s{i}") for i in range(3)]
        turns_success_2 = [make_turn(1, 0.1, False), make_turn(2, 0.9, True)]
        turns_success_4 = [
            make_turn(1, 0.1, False), make_turn(2, 0.2, False),
            make_turn(3, 0.3, False), make_turn(4, 0.9, True),
        ]
        results = [
            make_result(s1, is_success=True, turns=turns_success_2),
            make_result(s2, is_success=True, turns=turns_success_4),
            make_result(s3, is_success=False, turns=[]),  # excluded from mean
        ]
        report = RedTeamReport(model_name="test", results=results)
        assert abs(report.mean_turns_to_success - 3.0) < 1e-9  # (2+4)/2


# ── ModelComparison tests ─────────────────────────────────────────────────────


class TestModelComparison:
    def _make_report(self, model_name: str, n_success: int, n_total: int) -> RedTeamReport:
        results = []
        for i in range(n_total):
            s = make_scenario(f"s{i}")
            results.append(make_result(s, is_success=(i < n_success)))
        return RedTeamReport(model_name=model_name, results=results)

    def test_asr_delta_positive(self):
        baseline = self._make_report("base", n_success=2, n_total=10)   # 20%
        candidate = self._make_report("cand", n_success=4, n_total=10)  # 40%
        comp = ModelComparison(baseline=baseline, candidate=candidate)
        assert abs(comp.asr_delta - 0.2) < 1e-9

    def test_asr_delta_negative(self):
        baseline = self._make_report("base", n_success=5, n_total=10)   # 50%
        candidate = self._make_report("cand", n_success=2, n_total=10)  # 20%
        comp = ModelComparison(baseline=baseline, candidate=candidate)
        assert comp.asr_delta < 0

    def test_is_regression_true(self):
        """6pp increase → regression."""
        baseline = self._make_report("base", n_success=2, n_total=10)   # 20%
        candidate = self._make_report("cand", n_success=9, n_total=10)  # 90%
        comp = ModelComparison(baseline=baseline, candidate=candidate)
        assert comp.is_regression is True

    def test_is_regression_false_same(self):
        """Same ASR → no regression."""
        baseline = self._make_report("base", n_success=3, n_total=10)
        candidate = self._make_report("cand", n_success=3, n_total=10)
        comp = ModelComparison(baseline=baseline, candidate=candidate)
        assert comp.is_regression is False

    def test_is_regression_false_small_delta(self):
        """3pp increase → not a regression (< 5pp threshold)."""
        baseline = self._make_report("base", n_success=3, n_total=10)   # 30%
        candidate = self._make_report("cand", n_success=3, n_total=10)  # 30%
        comp = ModelComparison(baseline=baseline, candidate=candidate)
        assert comp.is_regression is False

    def test_category_deltas(self):
        """Category deltas should correctly track per-category changes."""
        cyber = make_scenario("c1", cat=HarmCategory.CYBER)
        chem = make_scenario("ch1", cat=HarmCategory.CHEMICAL)
        baseline = RedTeamReport("base", results=[
            make_result(cyber, is_success=False),
            make_result(chem, is_success=False),
        ])
        candidate = RedTeamReport("cand", results=[
            make_result(cyber, is_success=True),
            make_result(chem, is_success=False),
        ])
        comp = ModelComparison(baseline=baseline, candidate=candidate)
        deltas = comp.category_deltas()
        assert deltas[HarmCategory.CYBER.value] == 1.0
        assert deltas[HarmCategory.CHEMICAL.value] == 0.0

    def test_regression_summary_contains_models(self):
        baseline = self._make_report("baseline-model", n_success=2, n_total=10)
        candidate = self._make_report("candidate-model", n_success=8, n_total=10)
        comp = ModelComparison(baseline=baseline, candidate=candidate)
        summary = comp.regression_summary()
        assert "baseline-model" in summary
        assert "candidate-model" in summary
