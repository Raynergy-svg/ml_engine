"""Tests for LLM Macro Agent historical shadow validation script (US-006)."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.validate_llm_macro_shadow import (
    HISTORICAL_EVENTS,
    ValidationResult,
    generate_report,
    run_validation,
)


class TestHistoricalEvents:

    def test_events_list_not_empty(self):
        assert len(HISTORICAL_EVENTS) > 0

    def test_event_has_transcript(self):
        for ev in HISTORICAL_EVENTS:
            assert len(ev.pre_event_transcript) > 50

    def test_event_has_threshold(self):
        for ev in HISTORICAL_EVENTS:
            assert 0 < ev.expected_regime_shift <= 1.0


class TestRunValidation:

    def test_returns_results(self):
        results = run_validation()
        assert len(results) == len(HISTORICAL_EVENTS)
        for r in results:
            assert isinstance(r, ValidationResult)

    def test_result_has_required_fields(self):
        results = run_validation()
        for r in results:
            assert r.event_name
            assert r.event_date
            assert r.pair
            assert 0 <= r.regime_shift_probability <= 1
            assert r.direction_bias in ("LONG", "SHORT", "NEUTRAL")

    def test_passed_threshold_is_boolean(self):
        results = run_validation()
        for r in results:
            assert isinstance(r.passed_threshold, bool)


class TestGenerateReport:

    def test_report_contains_events(self):
        results = run_validation()
        md = generate_report(results)
        for r in results:
            assert r.event_name in md
        assert "Summary" in md

    def test_report_is_markdown(self):
        results = run_validation()
        md = generate_report(results)
        assert md.startswith("# LLM Macro Agent")
        assert "|" in md  # table


class TestScriptEntryPoint:

    def test_main_returns_zero(self):
        from scripts.validate_llm_macro_shadow import main
        assert main() == 0

    def test_report_files_created(self, tmp_path: Path):
        from scripts.validate_llm_macro_shadow import generate_report, run_validation
        results = run_validation()
        md = generate_report(results)
        # Just verify the report can be written
        f = tmp_path / "report.md"
        f.write_text(md)
        assert f.exists()
