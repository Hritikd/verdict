"""
Tests for the HarmBench scenario suite.

All CPU-only. Tests scenario filtering, ID lookup, severity ordering,
category coverage, and quick eval suite composition.
"""

from __future__ import annotations

import pytest

from verdict.core.types import HarmCategory, Severity
from verdict.scenarios.harmbench import (
    DEFAULT_SCENARIOS,
    get_quick_eval_suite,
    get_scenario_by_id,
    get_scenarios_by_category,
    get_scenarios_by_severity,
    scenarios_summary,
)


# ── Coverage tests ────────────────────────────────────────────────────────────


class TestDefaultScenarios:
    def test_at_least_20_scenarios(self):
        """Suite should have at least 20 scenarios to be meaningful."""
        assert len(DEFAULT_SCENARIOS) >= 20

    def test_all_have_ids(self):
        for s in DEFAULT_SCENARIOS:
            assert s.scenario_id, f"Scenario missing ID: {s}"
            assert len(s.scenario_id) > 0

    def test_all_have_goals(self):
        for s in DEFAULT_SCENARIOS:
            assert s.goal and len(s.goal) > 10, f"Scenario {s.scenario_id} has too-short goal"

    def test_all_have_target_behaviors(self):
        for s in DEFAULT_SCENARIOS:
            assert s.target_behavior and len(s.target_behavior) > 10, (
                f"Scenario {s.scenario_id} has too-short target_behavior"
            )

    def test_unique_ids(self):
        ids = [s.scenario_id for s in DEFAULT_SCENARIOS]
        assert len(set(ids)) == len(ids), "Duplicate scenario IDs found"

    def test_covers_cyber_category(self):
        cyber = [s for s in DEFAULT_SCENARIOS if s.category == HarmCategory.CYBER]
        assert len(cyber) >= 3, "Need at least 3 cyber scenarios"

    def test_covers_physical_category(self):
        physical = [s for s in DEFAULT_SCENARIOS if s.category == HarmCategory.PHYSICAL]
        assert len(physical) >= 2

    def test_covers_chemical_category(self):
        chemical = [s for s in DEFAULT_SCENARIOS if s.category == HarmCategory.CHEMICAL]
        assert len(chemical) >= 2

    def test_covers_harassment_category(self):
        harassment = [s for s in DEFAULT_SCENARIOS if s.category == HarmCategory.HARASSMENT]
        assert len(harassment) >= 2

    def test_covers_disinformation_category(self):
        disinfo = [s for s in DEFAULT_SCENARIOS if s.category == HarmCategory.DISINFORMATION]
        assert len(disinfo) >= 1

    def test_has_critical_severity_scenarios(self):
        critical = [s for s in DEFAULT_SCENARIOS if s.severity == Severity.CRITICAL]
        assert len(critical) >= 5, "Need critical-severity scenarios for strong evaluation"

    def test_has_high_severity_scenarios(self):
        high = [s for s in DEFAULT_SCENARIOS if s.severity == Severity.HIGH]
        assert len(high) >= 5


# ── get_scenarios_by_category tests ──────────────────────────────────────────


class TestGetByCategory:
    def test_cyber_filter(self):
        cyber = get_scenarios_by_category(HarmCategory.CYBER)
        assert all(s.category == HarmCategory.CYBER for s in cyber)
        assert len(cyber) > 0

    def test_chemical_filter(self):
        chemical = get_scenarios_by_category(HarmCategory.CHEMICAL)
        assert all(s.category == HarmCategory.CHEMICAL for s in chemical)

    def test_nonexistent_category_returns_empty(self):
        """ILLEGAL_DRUGS may not exist in scenarios — should return []."""
        drugs = get_scenarios_by_category(HarmCategory.ILLEGAL_DRUGS)
        # Either empty or all match
        assert all(s.category == HarmCategory.ILLEGAL_DRUGS for s in drugs)

    def test_returns_list(self):
        result = get_scenarios_by_category(HarmCategory.PHYSICAL)
        assert isinstance(result, list)


# ── get_scenarios_by_severity tests ──────────────────────────────────────────


class TestGetBySeverity:
    def test_critical_only(self):
        critical = get_scenarios_by_severity(Severity.CRITICAL, include_higher=False)
        assert all(s.severity == Severity.CRITICAL for s in critical)

    def test_high_and_above(self):
        high_plus = get_scenarios_by_severity(Severity.HIGH, include_higher=True)
        for s in high_plus:
            assert s.severity in (Severity.HIGH, Severity.CRITICAL)

    def test_medium_and_above(self):
        medium_plus = get_scenarios_by_severity(Severity.MEDIUM, include_higher=True)
        for s in medium_plus:
            assert s.severity in (Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL)

    def test_low_returns_all(self):
        """Low + above should return everything."""
        all_scenarios = get_scenarios_by_severity(Severity.LOW, include_higher=True)
        assert len(all_scenarios) == len(DEFAULT_SCENARIOS)

    def test_critical_subset_of_high_plus(self):
        critical = get_scenarios_by_severity(Severity.CRITICAL)
        high_plus = get_scenarios_by_severity(Severity.HIGH, include_higher=True)
        for s in critical:
            assert s in high_plus


# ── get_scenario_by_id tests ──────────────────────────────────────────────────


class TestGetScenarioById:
    def test_known_id_found(self):
        s = get_scenario_by_id("cyber_001")
        assert s is not None
        assert s.scenario_id == "cyber_001"

    def test_unknown_id_returns_none(self):
        s = get_scenario_by_id("nonexistent_999")
        assert s is None

    def test_all_default_ids_findable(self):
        for scenario in DEFAULT_SCENARIOS:
            found = get_scenario_by_id(scenario.scenario_id)
            assert found is not None
            assert found.scenario_id == scenario.scenario_id

    def test_id_case_sensitive(self):
        """IDs are case-sensitive strings."""
        upper = get_scenario_by_id("CYBER_001")
        assert upper is None  # IDs are lowercase


# ── get_quick_eval_suite tests ────────────────────────────────────────────────


class TestQuickEvalSuite:
    def test_has_exactly_5_scenarios(self):
        suite = get_quick_eval_suite()
        # Filter out None in case any IDs were changed
        suite = [s for s in suite if s is not None]
        assert len(suite) == 5

    def test_covers_multiple_categories(self):
        suite = [s for s in get_quick_eval_suite() if s is not None]
        categories = {s.category for s in suite}
        assert len(categories) >= 4, "Quick eval should cover ≥4 harm categories"

    def test_no_duplicate_scenarios(self):
        suite = [s for s in get_quick_eval_suite() if s is not None]
        ids = [s.scenario_id for s in suite]
        assert len(set(ids)) == len(ids)


# ── scenarios_summary tests ───────────────────────────────────────────────────


class TestScenariosSummary:
    def test_returns_dict(self):
        summary = scenarios_summary()
        assert isinstance(summary, dict)

    def test_counts_are_positive(self):
        summary = scenarios_summary()
        for cat, count in summary.items():
            assert count > 0, f"Category {cat} has count {count}"

    def test_total_equals_default_count(self):
        summary = scenarios_summary()
        assert sum(summary.values()) == len(DEFAULT_SCENARIOS)

    def test_cyber_in_summary(self):
        summary = scenarios_summary()
        assert HarmCategory.CYBER.value in summary
