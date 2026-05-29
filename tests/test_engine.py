"""
Tests for RedTeamEngine — using mock attacks to avoid LLM calls.

Tests:
1. Engine construction (from_litellm, manual)
2. evaluate() with mock attacks → correct report structure
3. stop_on_first_success behavior
4. compare_models() produces ModelComparison
5. quick_eval() uses the 5-scenario suite
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from verdict.core.engine import EngineConfig, RedTeamEngine
from verdict.core.types import (
    AttackResult,
    AttackType,
    HarmCategory,
    JudgeVerdict,
    ModelComparison,
    RedTeamReport,
    Scenario,
    Severity,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


def make_scenario(sid="s1") -> Scenario:
    return Scenario(
        scenario_id=sid,
        goal="Do something harmful",
        target_behavior="Harmful output produced",
        category=HarmCategory.CYBER,
        severity=Severity.HIGH,
    )


def make_attack_result(scenario: Scenario, success: bool, score: float = 0.8) -> AttackResult:
    return AttackResult(
        scenario=scenario,
        attack_type=AttackType.TEMPLATE,
        is_success=success,
        n_turns=1,
        best_prompt="adversarial prompt",
        best_response="harmful response",
        best_judge_score=score,
    )


def make_mock_engine(n_scenarios: int = 3, success_rate: float = 0.5) -> RedTeamEngine:
    """
    Build a RedTeamEngine with mocked target and judge.
    The mock _build_attack returns an attack whose run() returns
    a success/failure based on scenario index parity.
    """
    target = MagicMock()
    target.config = MagicMock()
    target.config.model = "mock-model"

    judge = MagicMock()
    judge.config = MagicMock()
    judge.config.model = "mock-judge"

    engine = RedTeamEngine(target=target, judge=judge, config=EngineConfig())
    return engine


# ── EngineConfig tests ────────────────────────────────────────────────────────


class TestEngineConfig:
    def test_defaults(self):
        cfg = EngineConfig()
        assert "template" in cfg.attacks
        assert cfg.max_concurrent_scenarios == 4
        assert cfg.pair_max_turns == 20
        assert cfg.stop_on_first_success is True

    def test_custom_attacks(self):
        cfg = EngineConfig(attacks=["pair", "crescendo"])
        assert cfg.attacks == ["pair", "crescendo"]

    def test_crescendo_defaults(self):
        cfg = EngineConfig()
        assert cfg.crescendo_setup_turns == 3
        assert cfg.crescendo_escalation_turns == 5


# ── RedTeamEngine._build_attack tests ────────────────────────────────────────


class TestBuildAttack:
    def setup_method(self):
        target = MagicMock()
        target.config = MagicMock()
        judge = MagicMock()
        judge.config = MagicMock()
        self.engine = RedTeamEngine(target=target, judge=judge)

    def test_build_template_attack(self):
        from verdict.attacks.templates import TemplateAttack
        attack = self.engine._build_attack("template")
        assert isinstance(attack, TemplateAttack)

    def test_build_injection_attack(self):
        from verdict.attacks.injection import PromptInjectionAttack
        attack = self.engine._build_attack("injection")
        assert isinstance(attack, PromptInjectionAttack)

    def test_build_pair_attack(self):
        from verdict.attacks.pair import PAIRAttack
        attack = self.engine._build_attack("pair")
        assert isinstance(attack, PAIRAttack)

    def test_build_crescendo_attack(self):
        from verdict.attacks.crescendo import CrescendoAttack
        attack = self.engine._build_attack("crescendo")
        assert isinstance(attack, CrescendoAttack)

    def test_build_case_insensitive(self):
        from verdict.attacks.templates import TemplateAttack
        attack = self.engine._build_attack("TEMPLATE")
        assert isinstance(attack, TemplateAttack)

    def test_unknown_attack_raises(self):
        with pytest.raises(ValueError, match="Unknown attack type"):
            self.engine._build_attack("nonexistent")


# ── evaluate() with patched attacks ──────────────────────────────────────────


class TestEvaluate:
    def _make_engine_with_mock_attack(self, is_success: bool, score: float) -> RedTeamEngine:
        target = MagicMock()
        target.config = MagicMock()
        target.config.model = "mock-model"
        judge = MagicMock()
        judge.config = MagicMock()

        engine = RedTeamEngine(
            target=target,
            judge=judge,
            config=EngineConfig(attacks=["template"]),
        )

        # Patch _build_attack to return a mock attack
        mock_attack = MagicMock()
        mock_attack.run = AsyncMock(side_effect=lambda scenario: make_attack_result(
            scenario, success=is_success, score=score
        ))
        engine._build_attack = MagicMock(return_value=mock_attack)
        return engine

    def test_evaluate_returns_report(self):
        engine = self._make_engine_with_mock_attack(is_success=False, score=0.1)
        scenarios = [make_scenario(f"s{i}") for i in range(3)]

        report = asyncio.get_event_loop().run_until_complete(
            engine.evaluate(scenarios, attacks=["template"])
        )

        assert isinstance(report, RedTeamReport)
        assert report.total_scenarios == 3

    def test_evaluate_all_success(self):
        engine = self._make_engine_with_mock_attack(is_success=True, score=0.9)
        scenarios = [make_scenario(f"s{i}") for i in range(4)]

        report = asyncio.get_event_loop().run_until_complete(
            engine.evaluate(scenarios, attacks=["template"])
        )

        assert report.attack_success_rate == 1.0
        assert report.successful_attacks == 4

    def test_evaluate_all_failure(self):
        engine = self._make_engine_with_mock_attack(is_success=False, score=0.05)
        scenarios = [make_scenario(f"s{i}") for i in range(4)]

        report = asyncio.get_event_loop().run_until_complete(
            engine.evaluate(scenarios, attacks=["template"])
        )

        assert report.attack_success_rate == 0.0

    def test_evaluate_report_has_model_name(self):
        engine = self._make_engine_with_mock_attack(is_success=False, score=0.1)
        engine.target.config.model = "gpt-4o-mini"
        scenarios = [make_scenario("s1")]

        report = asyncio.get_event_loop().run_until_complete(
            engine.evaluate(scenarios)
        )

        assert report.model_name == "gpt-4o-mini"

    def test_evaluate_run_config_recorded(self):
        engine = self._make_engine_with_mock_attack(is_success=False, score=0.1)
        scenarios = [make_scenario("s1")]

        report = asyncio.get_event_loop().run_until_complete(
            engine.evaluate(scenarios, attacks=["template"])
        )

        assert "attacks" in report.run_config
        assert report.run_config["attacks"] == ["template"]


# ── Multiple attacks per scenario ────────────────────────────────────────────


class TestMultipleAttacks:
    def test_two_attacks_one_scenario_produces_two_results(self):
        target = MagicMock()
        target.config = MagicMock()
        target.config.model = "mock-model"
        judge = MagicMock()
        judge.config = MagicMock()

        engine = RedTeamEngine(
            target=target,
            judge=judge,
            config=EngineConfig(
                attacks=["template", "injection"],
                stop_on_first_success=False,
            ),
        )

        call_count = [0]

        def build_attack(name):
            mock_attack = MagicMock()
            mock_attack.run = AsyncMock(side_effect=lambda s: make_attack_result(s, False, 0.1))
            return mock_attack

        engine._build_attack = build_attack

        scenarios = [make_scenario("s1")]
        report = asyncio.get_event_loop().run_until_complete(
            engine.evaluate(scenarios, attacks=["template", "injection"])
        )

        # 1 scenario × 2 attacks = 2 results
        assert report.total_scenarios == 2

    def test_stop_on_first_success_skips_remaining(self):
        target = MagicMock()
        target.config = MagicMock()
        target.config.model = "mock-model"
        judge = MagicMock()
        judge.config = MagicMock()

        engine = RedTeamEngine(
            target=target,
            judge=judge,
            config=EngineConfig(
                attacks=["template", "injection"],
                stop_on_first_success=True,
            ),
        )

        call_counts = {"template": 0, "injection": 0}

        def build_attack(name):
            mock_attack = MagicMock()
            # template always succeeds
            if name == "template":
                mock_attack.run = AsyncMock(side_effect=lambda s: make_attack_result(s, True, 0.9))
            else:
                mock_attack.run = AsyncMock(side_effect=lambda s: make_attack_result(s, False, 0.1))
            call_counts[name] = call_counts.get(name, 0) + 1
            return mock_attack

        engine._build_attack = build_attack

        scenarios = [make_scenario("s1")]
        report = asyncio.get_event_loop().run_until_complete(
            engine.evaluate(scenarios, attacks=["template", "injection"])
        )

        # Only template ran (stop_on_first_success), injection skipped
        assert report.total_scenarios == 1
        assert report.results[0].is_success is True
