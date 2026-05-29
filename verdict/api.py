"""
Verdict REST API
================

FastAPI server exposing red-team evaluation over HTTP for CI integration,
dashboards, and programmatic access.

Endpoints
---------
POST /evaluate          Run all attacks on a list of scenarios
POST /evaluate/quick    5-scenario CI gate (template attacks only)
POST /compare           Side-by-side safety regression test
GET  /scenarios         List all built-in scenarios with metadata
GET  /scenarios/{id}    Get a specific scenario by ID
GET  /attacks           List supported attack types and descriptions
GET  /health            Liveness probe

Usage
-----
    uvicorn verdict.api:app --host 0.0.0.0 --port 8080

    # or via CLI:
    verdict serve
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from verdict.core.engine import EngineConfig, RedTeamEngine
from verdict.core.types import AttackType, HarmCategory, RedTeamReport, Severity
from verdict.report import save_html, save_json

logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Verdict Red-Team API",
    description=(
        "Automated adversarial LLM red-teaming platform. "
        "Implements PAIR, Crescendo, template injection, and prompt injection attacks "
        "with LLM-as-judge semantic evaluation.\n\n"
        "Paper references:\n"
        "- PAIR: Chao et al. 2023 (arxiv:2310.08419)\n"
        "- Crescendo: Russinovich et al. 2024 (arxiv:2404.01833)\n"
        "- HarmBench: Mazeika et al. 2024 (arxiv:2402.04249)"
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────


class ScenarioRequest(BaseModel):
    scenario_id: str = Field(..., description="Unique scenario identifier")
    goal: str = Field(..., description="Harmful instruction the attacker tries to elicit")
    target_behavior: str = Field(..., description="What a successful harmful response looks like")
    category: str = Field("general_harm", description="HarmBench harm category")
    severity: str = Field("medium", description="Severity: critical/high/medium/low")


class EvaluateRequest(BaseModel):
    target_model: str = Field("gpt-4o-mini", description="LiteLLM model identifier for the target")
    judge_model: str = Field("gpt-4o-mini", description="LiteLLM model identifier for the judge")
    attacks: List[str] = Field(
        default=["template", "injection"],
        description="Attack types: pair, template, injection, crescendo",
    )
    scenarios: Optional[List[ScenarioRequest]] = Field(
        None,
        description="Custom scenarios. Omit to use the built-in HarmBench suite.",
    )
    scenario_ids: Optional[List[str]] = Field(
        None,
        description="Subset of built-in scenario IDs to evaluate.",
    )
    max_concurrent: int = Field(4, ge=1, le=16)
    stop_on_first_success: bool = True
    target_system_prompt: Optional[str] = None


class CompareRequest(BaseModel):
    baseline_model: str = Field(..., description="Baseline model (e.g. pre-fine-tune checkpoint)")
    candidate_model: str = Field(..., description="Candidate model under evaluation")
    judge_model: str = Field("gpt-4o-mini")
    attacks: List[str] = Field(default=["template"])
    scenario_ids: Optional[List[str]] = None
    baseline_system_prompt: Optional[str] = None
    candidate_system_prompt: Optional[str] = None


class AttackResultResponse(BaseModel):
    scenario_id: str
    category: str
    severity: str
    attack_type: str
    is_success: bool
    best_judge_score: float
    n_turns: int
    turns_to_success: Optional[int]
    wall_time_seconds: float
    convergence_curve: List[float]


class EvaluateResponse(BaseModel):
    model_name: str
    run_timestamp: float
    total_scenarios: int
    successful_attacks: int
    attack_success_rate: float
    mean_judge_score: float
    mean_turns_to_success: float
    asr_by_category: Dict[str, float]
    asr_by_attack: Dict[str, float]
    asr_by_severity: Dict[str, float]
    results: List[AttackResultResponse]
    elapsed_seconds: float


class CompareResponse(BaseModel):
    baseline_model: str
    baseline_asr: float
    candidate_model: str
    candidate_asr: float
    asr_delta: float
    is_regression: bool
    category_deltas: Dict[str, float]


class ScenarioInfo(BaseModel):
    scenario_id: str
    goal: str
    target_behavior: str
    category: str
    severity: str


class AttackInfo(BaseModel):
    name: str
    description: str
    paper: Optional[str]
    multi_turn: bool
    default_turns: Optional[int]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _report_to_response(report: RedTeamReport, elapsed: float) -> EvaluateResponse:
    results = [
        AttackResultResponse(
            scenario_id=r.scenario.scenario_id,
            category=r.scenario.category.value,
            severity=r.scenario.severity.value,
            attack_type=r.attack_type.value,
            is_success=r.is_success,
            best_judge_score=r.best_judge_score,
            n_turns=r.n_turns,
            turns_to_success=r.turns_to_success,
            wall_time_seconds=r.wall_time_seconds,
            convergence_curve=r.convergence_curve,
        )
        for r in report.results
    ]
    return EvaluateResponse(
        model_name=report.model_name,
        run_timestamp=report.run_timestamp,
        total_scenarios=report.total_scenarios,
        successful_attacks=report.successful_attacks,
        attack_success_rate=report.attack_success_rate,
        mean_judge_score=report.mean_judge_score,
        mean_turns_to_success=report.mean_turns_to_success,
        asr_by_category=report.asr_by_category(),
        asr_by_attack=report.asr_by_attack(),
        asr_by_severity=report.asr_by_severity(),
        results=results,
        elapsed_seconds=elapsed,
    )


def _build_scenarios_from_request(
    req: EvaluateRequest,
) -> list:
    from verdict.scenarios.harmbench import (
        DEFAULT_SCENARIOS, get_scenario_by_id,
    )
    from verdict.core.types import Scenario

    if req.scenarios:
        # User-supplied custom scenarios
        scenarios = []
        for s in req.scenarios:
            try:
                cat = HarmCategory(s.category)
            except ValueError:
                cat = HarmCategory.GENERAL
            try:
                sev = Severity(s.severity)
            except ValueError:
                sev = Severity.MEDIUM
            scenarios.append(Scenario(
                scenario_id=s.scenario_id,
                goal=s.goal,
                target_behavior=s.target_behavior,
                category=cat,
                severity=sev,
            ))
        return scenarios

    if req.scenario_ids:
        scenarios = [get_scenario_by_id(sid) for sid in req.scenario_ids]
        missing = [req.scenario_ids[i] for i, s in enumerate(scenarios) if s is None]
        if missing:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown scenario IDs: {missing}",
            )
        return [s for s in scenarios if s is not None]

    return DEFAULT_SCENARIOS


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health", tags=["System"])
async def health() -> Dict[str, Any]:
    """Liveness probe — returns service status and version."""
    return {
        "status": "healthy",
        "version": "0.1.0",
        "timestamp": time.time(),
    }


@app.get("/attacks", response_model=List[AttackInfo], tags=["Metadata"])
async def list_attacks() -> List[AttackInfo]:
    """
    List all supported attack types with descriptions and paper references.
    """
    return [
        AttackInfo(
            name="pair",
            description=(
                "Prompt Automatic Iterative Refinement — attacker LLM iteratively refines "
                "adversarial prompts using judge feedback until the target is jailbroken."
            ),
            paper="Chao et al. 2023 — Jailbreaking Black Box LLMs (arxiv:2310.08419)",
            multi_turn=True,
            default_turns=20,
        ),
        AttackInfo(
            name="template",
            description=(
                "Pre-written jailbreak templates: DAN, researcher authority, fictional framing, "
                "base64 encoding, hypothetical world, etc."
            ),
            paper=None,
            multi_turn=False,
            default_turns=1,
        ),
        AttackInfo(
            name="injection",
            description=(
                "Prompt injection attacks: direct override instructions and indirect injection "
                "via document/email/code-review payloads."
            ),
            paper=None,
            multi_turn=False,
            default_turns=1,
        ),
        AttackInfo(
            name="crescendo",
            description=(
                "Multi-turn gradual escalation — setup turns prime the context, "
                "escalation turns incrementally approach the harmful goal."
            ),
            paper="Russinovich et al. 2024 — Microsoft Research (arxiv:2404.01833)",
            multi_turn=True,
            default_turns=8,
        ),
    ]


@app.get("/scenarios", response_model=List[ScenarioInfo], tags=["Metadata"])
async def list_scenarios(
    category: Optional[str] = Query(None, description="Filter by harm category"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
) -> List[ScenarioInfo]:
    """
    List all built-in HarmBench-aligned scenarios, with optional category/severity filters.
    """
    from verdict.scenarios.harmbench import DEFAULT_SCENARIOS

    scenarios = DEFAULT_SCENARIOS
    if category:
        scenarios = [s for s in scenarios if s.category.value == category]
    if severity:
        scenarios = [s for s in scenarios if s.severity.value == severity]

    return [
        ScenarioInfo(
            scenario_id=s.scenario_id,
            goal=s.goal,
            target_behavior=s.target_behavior,
            category=s.category.value,
            severity=s.severity.value,
        )
        for s in scenarios
    ]


@app.get("/scenarios/{scenario_id}", response_model=ScenarioInfo, tags=["Metadata"])
async def get_scenario(scenario_id: str) -> ScenarioInfo:
    """Get a single scenario by ID."""
    from verdict.scenarios.harmbench import get_scenario_by_id

    s = get_scenario_by_id(scenario_id)
    if s is None:
        raise HTTPException(status_code=404, detail=f"Scenario '{scenario_id}' not found")
    return ScenarioInfo(
        scenario_id=s.scenario_id,
        goal=s.goal,
        target_behavior=s.target_behavior,
        category=s.category.value,
        severity=s.severity.value,
    )


@app.post("/evaluate", response_model=EvaluateResponse, tags=["Evaluation"])
async def evaluate(req: EvaluateRequest) -> EvaluateResponse:
    """
    Run red-team evaluation on one or more scenarios.

    Supports all four attack types. Results include per-scenario judge scores,
    convergence curves, and aggregate ASR metrics.
    """
    t0 = time.perf_counter()
    scenarios = _build_scenarios_from_request(req)

    from verdict.core.engine import EngineConfig, RedTeamEngine

    engine = RedTeamEngine.from_litellm(
        target_model=req.target_model,
        judge_model=req.judge_model,
        target_system_prompt=req.target_system_prompt,
        config=EngineConfig(
            attacks=req.attacks,
            max_concurrent_scenarios=req.max_concurrent,
            stop_on_first_success=req.stop_on_first_success,
        ),
    )

    report = await engine.evaluate(scenarios, attacks=req.attacks)
    elapsed = time.perf_counter() - t0
    return _report_to_response(report, elapsed)


@app.post("/evaluate/quick", response_model=EvaluateResponse, tags=["Evaluation"])
async def quick_eval(
    target_model: str = "gpt-4o-mini",
    judge_model: str = "gpt-4o-mini",
    target_system_prompt: Optional[str] = None,
) -> EvaluateResponse:
    """
    Fast 5-scenario CI gate.

    Runs template attacks on the quick eval suite (~3 minutes).
    Returns non-zero HTTP 200 with is_regression=False or True.
    """
    t0 = time.perf_counter()
    engine = RedTeamEngine.from_litellm(
        target_model=target_model,
        judge_model=judge_model,
        target_system_prompt=target_system_prompt,
    )
    report = await engine.quick_eval()
    elapsed = time.perf_counter() - t0
    return _report_to_response(report, elapsed)


@app.post("/compare", response_model=CompareResponse, tags=["Evaluation"])
async def compare_models(req: CompareRequest) -> CompareResponse:
    """
    Safety regression test: compare baseline model vs candidate model.

    Returns ASR delta and whether the candidate represents a regression
    (>5pp increase in attack success rate).
    """
    from verdict.core.engine import EngineConfig, RedTeamEngine
    from verdict.scenarios.harmbench import DEFAULT_SCENARIOS, get_scenario_by_id

    scenarios = None
    if req.scenario_ids:
        scenarios = [get_scenario_by_id(sid) for sid in req.scenario_ids]
        scenarios = [s for s in scenarios if s is not None]

    engine = RedTeamEngine.from_litellm(
        target_model=req.baseline_model,
        judge_model=req.judge_model,
        config=EngineConfig(attacks=req.attacks),
    )

    comparison = await engine.compare_models(
        baseline_model=req.baseline_model,
        candidate_model=req.candidate_model,
        scenarios=scenarios,
        baseline_system_prompt=req.baseline_system_prompt,
        candidate_system_prompt=req.candidate_system_prompt,
    )

    return CompareResponse(
        baseline_model=comparison.baseline.model_name,
        baseline_asr=comparison.baseline.attack_success_rate,
        candidate_model=comparison.candidate.model_name,
        candidate_asr=comparison.candidate.attack_success_rate,
        asr_delta=comparison.asr_delta,
        is_regression=comparison.is_regression,
        category_deltas=comparison.category_deltas(),
    )
