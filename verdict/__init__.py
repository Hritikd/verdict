"""
Verdict — Automated Adversarial LLM Red-Teaming Platform
=========================================================

Verdict implements four state-of-the-art jailbreak attack strategies
(PAIR, Crescendo, template injection, prompt injection) against an
LLM-as-judge evaluation harness, producing ASR metrics, per-category
breakdowns, and visual HTML reports.

Quick start
-----------
    from verdict import RedTeamEngine
    from verdict.scenarios import DEFAULT_SCENARIOS

    engine = RedTeamEngine.from_litellm(
        target_model="gpt-4o-mini",
        judge_model="gpt-4o",
    )
    report = await engine.evaluate(DEFAULT_SCENARIOS[:5], attacks=["template"])
    print(report.summary_table())

References
----------
- HarmBench (Mazeika et al., 2024): https://arxiv.org/abs/2402.04249
- PAIR (Chao et al., 2023): https://arxiv.org/abs/2310.08419
- Crescendo (Russinovich et al., 2024): https://arxiv.org/abs/2404.01833
"""

from verdict.core.engine import EngineConfig, RedTeamEngine
from verdict.core.judge import JudgeConfig, SemanticJudge
from verdict.core.target import TargetConfig, TargetModel
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

__version__ = "0.1.0"
__author__ = "Hritik Datta"

__all__ = [
    # Engine
    "RedTeamEngine",
    "EngineConfig",
    # Judge
    "SemanticJudge",
    "JudgeConfig",
    # Target
    "TargetModel",
    "TargetConfig",
    # Types
    "AttackResult",
    "AttackType",
    "HarmCategory",
    "JudgeVerdict",
    "ModelComparison",
    "RedTeamReport",
    "Scenario",
    "Severity",
]
