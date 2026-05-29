#!/usr/bin/env python3
"""
Run a full red-team evaluation and save JSON + HTML reports.

Usage:
    python scripts/run_evaluation.py --model gpt-4o-mini --attacks template injection

This script is a convenience wrapper around the Verdict engine for
running evaluations outside of the CLI or API server.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

# Ensure the package is importable when run from project root
sys.path.insert(0, str(Path(__file__).parent.parent))


async def main():
    parser = argparse.ArgumentParser(
        description="Run Verdict red-team evaluation"
    )
    parser.add_argument(
        "--model", "-m", default="gpt-4o-mini",
        help="Target model (LiteLLM identifier, e.g. gpt-4o-mini, claude-3-haiku-20240307)",
    )
    parser.add_argument(
        "--judge", default="gpt-4o-mini",
        help="Judge model",
    )
    parser.add_argument(
        "--attacks", "-a", default="template,injection",
        help="Comma-separated attack list: pair,template,injection,crescendo",
    )
    parser.add_argument(
        "--scenarios", "-s", default=None,
        help="Comma-separated scenario IDs. Default: all 20 built-in scenarios.",
    )
    parser.add_argument(
        "--output-dir", "-o", default="reports",
        help="Output directory for JSON and HTML reports",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run quick 5-scenario eval instead of full suite",
    )
    parser.add_argument(
        "--system-prompt", default=None,
        help="System prompt for the target model",
    )
    args = parser.parse_args()

    from verdict.core.engine import EngineConfig, RedTeamEngine
    from verdict.report import save_html, save_json
    from verdict.scenarios.harmbench import DEFAULT_SCENARIOS, get_scenario_by_id

    attack_list = [a.strip() for a in args.attacks.split(",")]

    engine = RedTeamEngine.from_litellm(
        target_model=args.model,
        judge_model=args.judge,
        target_system_prompt=args.system_prompt,
        config=EngineConfig(attacks=attack_list),
    )

    if args.quick:
        print(f"Running quick eval on {args.model}...")
        report = await engine.quick_eval()
    else:
        if args.scenarios:
            scenarios = [get_scenario_by_id(sid.strip()) for sid in args.scenarios.split(",")]
            scenarios = [s for s in scenarios if s is not None]
        else:
            scenarios = DEFAULT_SCENARIOS

        print(
            f"Running evaluation: {len(scenarios)} scenarios × {attack_list} "
            f"attacks on {args.model}..."
        )
        t0 = time.perf_counter()
        report = await engine.evaluate(scenarios, attacks=attack_list)
        elapsed = time.perf_counter() - t0
        print(f"Completed in {elapsed:.1f}s")

    # Print summary
    print(report.summary_table())

    # Save reports
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_slug = args.model.replace("/", "_").replace(":", "_")
    timestamp = int(time.time())

    json_path = save_json(report, str(out_dir / f"{model_slug}_{timestamp}.json"))
    html_path = save_html(report, str(out_dir / f"{model_slug}_{timestamp}.html"))

    print(f"\nReports saved:")
    print(f"  JSON: {json_path}")
    if html_path:
        print(f"  HTML: {html_path}")


if __name__ == "__main__":
    asyncio.run(main())
