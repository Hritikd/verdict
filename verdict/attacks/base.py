"""Base class for all attack implementations."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
from verdict.core.types import AttackResult, AttackType, Scenario
from verdict.core.judge import SemanticJudge
from verdict.core.target import TargetModel


class BaseAttack(ABC):
    """Abstract base for all attack types."""

    attack_type: AttackType

    def __init__(self, judge: SemanticJudge, target: TargetModel) -> None:
        self.judge = judge
        self.target = target

    @abstractmethod
    async def run(self, scenario: Scenario, **kwargs) -> AttackResult:
        """Execute the attack against the target for the given scenario."""
        ...

    def _make_failed_result(self, scenario: Scenario, n_turns: int = 0) -> AttackResult:
        return AttackResult(
            scenario=scenario,
            attack_type=self.attack_type,
            is_success=False,
            n_turns=n_turns,
            best_prompt="",
            best_response="",
            best_judge_score=0.0,
            model_name=self.target.config.model,
        )
