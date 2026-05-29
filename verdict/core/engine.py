"""
RedTeamEngine — Orchestrates multi-attack, multi-scenario evaluations.

The engine is the central coordinator:
1. Accepts a list of scenarios and attack configurations
2. Runs each attack × scenario combination
3. Aggregates results into a RedTeamReport
4. Supports model regression testing: compare baseline vs. candidate

Usage
-----
    engine = RedTeamEngine.from_litellm(
        target_model="gpt-4o-mini",
        judge_model="gpt-4o",
    )

    report = await engine.evaluate(
        scenarios=DEFAULT_SCENARIOS,
        attacks=["pair", "template", "injection"],
    )
    print(report.summary_table())

    # Regression test: did fine-tuning break safety?
    comparison = await engine.compare_models(
        baseline_model="gpt-4o-mini",
        candidate_model="my-fine-tuned-model",
        scenarios=get_quick_eval_suite(),
    )
    print(comparison.regression_summary())
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from verdict.core.judge import JudgeConfig, SemanticJudge
from verdict.core.target import TargetConfig, TargetModel
from verdict.core.types import (
    AttackResult,
    AttackType,
    ModelComparison,
    RedTeamReport,
    Scenario,
)

logger = logging.getLogger(__name__)


@dataclass
class EngineConfig:
    """Configuration for the red-team engine."""

    attacks: List[str] = field(
        default_factory=lambda: ["template", "injection", "pair"]
    )
    """Attack types to run. Options: 'pair', 'template', 'injection', 'crescendo'."""

    max_concurrent_scenarios: int = 4
    """Max scenarios evaluated in parallel."""

    pair_max_turns: int = 20
    pair_n_streams: int = 1

    crescendo_setup_turns: int = 3
    crescendo_escalation_turns: int = 5

    stop_on_first_success: bool = True
    """If True, skip remaining attacks for a scenario once one succeeds."""


class RedTeamEngine:
    """
    Orchestrates red-team evaluations across multiple scenarios and attacks.
    """

    def __init__(
        self,
        target: TargetModel,
        judge: SemanticJudge,
        config: Optional[EngineConfig] = None,
    ) -> None:
        self.target = target
        self.judge = judge
        self.config = config or EngineConfig()

    # ── Constructors ─────────────────────────────────────────────────────────

    @classmethod
    def from_litellm(
        cls,
        target_model: str = "gpt-4o-mini",
        judge_model: str = "gpt-4o-mini",
        target_system_prompt: Optional[str] = None,
        config: Optional[EngineConfig] = None,
    ) -> "RedTeamEngine":
        target = TargetModel(TargetConfig(
            model=target_model,
            system_prompt=target_system_prompt,
        ))
        judge = SemanticJudge(JudgeConfig(model=judge_model))
        return cls(target, judge, config)

    # ── Evaluation ───────────────────────────────────────────────────────────

    async def evaluate(
        self,
        scenarios: List[Scenario],
        attacks: Optional[List[str]] = None,
    ) -> RedTeamReport:
        """
        Run all specified attacks on all scenarios.

        Parameters
        ----------
        scenarios:
            List of Scenario objects to evaluate against.
        attacks:
            Attack types to run (overrides config). None = use config.

        Returns
        -------
        RedTeamReport with per-scenario and aggregate results.
        """
        attack_names = attacks or self.config.attacks
        t0 = time.perf_counter()

        logger.info(
            "Starting evaluation: %d scenarios × %d attacks = %d combinations",
            len(scenarios), len(attack_names), len(scenarios) * len(attack_names),
        )

        sem = asyncio.Semaphore(self.config.max_concurrent_scenarios)
        all_results: List[AttackResult] = []

        async def _evaluate_scenario(scenario: Scenario) -> List[AttackResult]:
            async with sem:
                return await self._run_attacks_on_scenario(scenario, attack_names)

        scenario_results = await asyncio.gather(
            *[_evaluate_scenario(s) for s in scenarios]
        )

        for result_list in scenario_results:
            all_results.extend(result_list)

        report = RedTeamReport(
            model_name=self.target.config.model,
            results=all_results,
            run_config={
                "attacks": attack_names,
                "n_scenarios": len(scenarios),
                "target_model": self.target.config.model,
                "judge_model": self.judge.config.model,
            },
        )

        logger.info(
            "Evaluation complete: ASR=%.1f%% (%d/%d) wall=%.1fs",
            report.attack_success_rate * 100,
            report.successful_attacks,
            report.total_scenarios,
            time.perf_counter() - t0,
        )

        return report

    async def _run_attacks_on_scenario(
        self,
        scenario: Scenario,
        attack_names: List[str],
    ) -> List[AttackResult]:
        """Run all attacks on a single scenario."""
        results = []
        for attack_name in attack_names:
            attack = self._build_attack(attack_name)
            result = await attack.run(scenario)
            results.append(result)
            logger.debug(
                "  %s × %s: success=%s score=%.3f",
                attack_name, scenario.scenario_id,
                result.is_success, result.best_judge_score,
            )
            # Skip remaining attacks if this one succeeded and stop_on_first_success
            if result.is_success and self.config.stop_on_first_success:
                # Fill remaining attack slots with "skipped" results
                break

        return results

    def _build_attack(self, attack_name: str):
        """Instantiate an attack object by name."""
        from verdict.attacks.pair import PAIRAttack, PAIRConfig
        from verdict.attacks.templates import TemplateAttack, TemplateConfig
        from verdict.attacks.injection import PromptInjectionAttack, InjectionConfig
        from verdict.attacks.crescendo import CrescendoAttack, CrescendoConfig

        name = attack_name.lower()
        if name == "pair":
            return PAIRAttack(self.judge, self.target, PAIRConfig(
                max_turns=self.config.pair_max_turns,
                n_concurrent_streams=self.config.pair_n_streams,
            ))
        elif name == "template":
            return TemplateAttack(self.judge, self.target, TemplateConfig(
                stop_on_first_success=self.config.stop_on_first_success,
            ))
        elif name == "injection":
            return PromptInjectionAttack(self.judge, self.target, InjectionConfig(
                stop_on_first_success=self.config.stop_on_first_success,
            ))
        elif name == "crescendo":
            return CrescendoAttack(self.judge, self.target, CrescendoConfig(
                n_setup_turns=self.config.crescendo_setup_turns,
                n_escalation_turns=self.config.crescendo_escalation_turns,
            ))
        else:
            raise ValueError(f"Unknown attack type: {attack_name!r}. "
                             f"Valid: pair, template, injection, crescendo")

    # ── Model comparison / regression testing ────────────────────────────────

    async def compare_models(
        self,
        baseline_model: str,
        candidate_model: str,
        scenarios: Optional[List[Scenario]] = None,
        baseline_system_prompt: Optional[str] = None,
        candidate_system_prompt: Optional[str] = None,
    ) -> ModelComparison:
        """
        Run the same evaluation against two models and compare safety.

        Used for regression testing: did the new fine-tune break safety
        properties that the base model had?

        Parameters
        ----------
        baseline_model:
            The model to compare against (e.g. the pre-fine-tune checkpoint).
        candidate_model:
            The model being evaluated (e.g. post-fine-tune).
        scenarios:
            Scenarios to test. Defaults to DEFAULT_SCENARIOS.
        """
        from verdict.scenarios.harmbench import DEFAULT_SCENARIOS
        eval_scenarios = scenarios or DEFAULT_SCENARIOS[:10]  # use first 10 by default

        logger.info(
            "Model comparison: %s vs %s on %d scenarios",
            baseline_model, candidate_model, len(eval_scenarios),
        )

        # Evaluate baseline
        baseline_target = TargetModel(TargetConfig(
            model=baseline_model,
            system_prompt=baseline_system_prompt,
        ))
        baseline_engine = RedTeamEngine(baseline_target, self.judge, self.config)
        baseline_report = await baseline_engine.evaluate(eval_scenarios)

        # Evaluate candidate
        candidate_target = TargetModel(TargetConfig(
            model=candidate_model,
            system_prompt=candidate_system_prompt,
        ))
        candidate_engine = RedTeamEngine(candidate_target, self.judge, self.config)
        candidate_report = await candidate_engine.evaluate(eval_scenarios)

        comparison = ModelComparison(baseline=baseline_report, candidate=candidate_report)

        logger.info(
            "Comparison: %s ASR=%.1f%% | %s ASR=%.1f%% | delta=%+.1f%% | regression=%s",
            baseline_model, baseline_report.attack_success_rate * 100,
            candidate_model, candidate_report.attack_success_rate * 100,
            comparison.asr_delta * 100,
            comparison.is_regression,
        )

        return comparison

    # ── Quick evaluation ──────────────────────────────────────────────────────

    async def quick_eval(self) -> RedTeamReport:
        """
        Fast 5-scenario evaluation for CI integration.

        Runs template attacks only on the quick eval suite (~3 minutes).
        Suitable for a go/no-go safety gate in a deployment pipeline.
        """
        from verdict.scenarios.harmbench import get_quick_eval_suite
        scenarios = [s for s in get_quick_eval_suite() if s is not None]
        return await self.evaluate(scenarios, attacks=["template"])
