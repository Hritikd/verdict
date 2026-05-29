#!/usr/bin/env python3
"""
Safety regression test: compare two models' ASR on the same scenario suite.

Usage:
    python scripts/compare_models.py \
        --baseline gpt-3.5-turbo \
        --candidate my-fine-tuned-model \
        --attacks template \
        --fail-on-regression

Exit codes:
    0 — no regression (or --fail-on-regression not set)
    1 — regression detected (candidate ASR > baseline ASR + 5pp)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


async def main() -> int:
    parser = argparse.ArgumentParser(description="Verdict model safety comparison")
    parser.add_argument("--baseline", "-b", required=True,
                        help="Baseline model identifier")
    parser.add_argument("--candidate", "-c", required=True,
                        help="Candidate model identifier")
    parser.add_argument("--judge", default="gpt-4o-mini")
    parser.add_argument("--attacks", "-a", default="template",
                        help="Comma-separated attack types")
    parser.add_argument("--scenarios", "-s", default=None,
                        help="Comma-separated scenario IDs (default: first 10)")
    parser.add_argument("--output-json", "-o", default=None,
                        help="Save comparison JSON to this path")
    parser.add_argument("--fail-on-regression", action="store_true",
                        help="Exit with code 1 if regression detected")
    args = parser.parse_args()

    from verdict.core.engine import EngineConfig, RedTeamEngine
    from verdict.report import save_comparison_json
    from verdict.scenarios.harmbench import DEFAULT_SCENARIOS, get_scenario_by_id

    attack_list = [a.strip() for a in args.attacks.split(",")]

    scenario_list = None
    if args.scenarios:
        scenario_list = [
            get_scenario_by_id(sid.strip())
            for sid in args.scenarios.split(",")
        ]
        scenario_list = [s for s in scenario_list if s is not None]

    print(
        f"Comparing:\n"
        f"  Baseline:  {args.baseline}\n"
        f"  Candidate: {args.candidate}\n"
        f"  Attacks:   {attack_list}\n"
    )

    engine = RedTeamEngine.from_litellm(
        target_model=args.baseline,
        judge_model=args.judge,
        config=EngineConfig(attacks=attack_list),
    )

    comparison = await engine.compare_models(
        baseline_model=args.baseline,
        candidate_model=args.candidate,
        scenarios=scenario_list,
    )

    print(comparison.regression_summary())

    if args.output_json:
        path = save_comparison_json(comparison, args.output_json)
        print(f"\nComparison saved to: {path}")

    if args.fail_on_regression and comparison.is_regression:
        print(
            f"\nREGRESSION DETECTED: Δ ASR = {comparison.asr_delta:+.1%}\n"
            f"Candidate {args.candidate} is less safe than baseline {args.baseline}."
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
