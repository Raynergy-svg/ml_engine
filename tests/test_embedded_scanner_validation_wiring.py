"""Regression for US-604 TUI wiring gap (discovered 2026-04-25 during US-606 close-out).

Bug class: ``EmbeddedScanner`` (the TUI's scanner, used by ``./buddy``)
cherry-picks automation modules from ``ContinuousScanner`` rather than
subclassing it. So when wiring is added to ``continuous.py`` only, it is
silently dead in the TUI scan loop. Symptom: ``dry_run_validation.jsonl``
accumulates ZERO rows during the 48h validation window mandated by Phase 95
US-604, and the LIVE re-enable gate cannot complete.

This is the same fingerprint as the 2026-04-16 ConfigAdjuster orphan-key
incident ($3,527 production loss) — same architectural seam, same
asymmetric-wiring failure mode.

These tests lock in the symmetry. If anyone ever removes the
``validation_stats`` wiring from ``embedded_scanner._post_scan_automation``,
this suite fails loudly.

Phase 96 candidate:  hoist all automation wiring into a shared
``automation_pipeline.py`` so neither scanner can drift. Until that lands,
this test is the guardrail.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Static guarantees — symbols are present in both scanners
# ---------------------------------------------------------------------------

class TestStaticWiring:
    def test_validation_stats_imported_in_embedded_scanner(self) -> None:
        """The string 'validation_stats' must appear in embedded_scanner.py.

        Cheap, lints fast, catches accidental removal of the import line.
        """
        text = Path("src/tui/embedded_scanner.py").read_text()
        assert "validation_stats" in text, (
            "embedded_scanner.py must import validation_stats — "
            "see docs/supervisor_console_runbook.md §12.4 for the patch."
        )
        assert "ScanDistributionStats" in text, (
            "embedded_scanner.py must reference ScanDistributionStats."
        )

    def test_record_cycle_called_in_post_scan_automation(self) -> None:
        """The actual call site must live in _post_scan_automation.

        Reading the source for ``record_cycle(`` ensures the wiring is *active*
        — not just imported and forgotten.
        """
        from src.tui.embedded_scanner import EmbeddedScanner

        method = EmbeddedScanner._post_scan_automation
        src = inspect.getsource(method)
        assert "record_cycle" in src, (
            "_post_scan_automation must call _validation_stats.record_cycle(...)"
        )

    def test_continuous_scanner_still_has_wiring(self) -> None:
        """ContinuousScanner kept its wiring — symmetry holds."""
        text = Path("src/scanner/automation/continuous.py").read_text()
        assert "validation_stats" in text
        assert "record_cycle" in text


# ---------------------------------------------------------------------------
# Behavioral guarantee — wiring actually emits rows when invoked
# ---------------------------------------------------------------------------

class TestEmbeddedScannerEmitsRows:
    def test_post_scan_automation_writes_jsonl_row(
        self,
        tmp_path: Path,
    ) -> None:
        """End-to-end: when _post_scan_automation runs against a result with
        directional candidates, a row appears in dry_run_validation.jsonl.

        Strategy: pre-create the _validation_stats instance pointed at tmp_path
        so the wiring's ``if not hasattr(...)`` check uses our test instance.
        Avoids monkey-patching the dataclass default (which is captured at
        class definition and cannot be overridden post hoc).
        """
        from src.scanner.automation.validation_stats import ScanDistributionStats
        from src.tui.embedded_scanner import EmbeddedScanner

        out = tmp_path / "dry_run_validation.jsonl"

        # Plain MagicMock (no spec) so any attribute access in
        # _post_scan_automation auto-resolves rather than AttributeError-ing.
        es = MagicMock()
        es._scan_count = 1
        es._brain = lambda *a, **kw: None
        es._scanner = MagicMock()
        es._scanner._observation_consumer = None
        es._online_rl = None
        es._config_adjuster = None
        es._config = None
        es._model_freshness = None
        # Pre-create the validation_stats instance pointed at tmp_path
        es._validation_stats = ScanDistributionStats(log_path=out, enabled=True)

        # Build a fake scan result with one directional candidate
        candidate = MagicMock()
        candidate.pair = "EUR_USD"
        candidate.direction = "LONG"
        candidate.confidence = 0.65
        candidate.weighted_vote_score = 0.72
        candidate.model_disagreement = 0.20
        candidate.volatility_regime = "NORMAL"
        candidate.gates_passed = True
        candidate.is_tradeable = True
        candidate.error = None

        result = MagicMock()
        result.analyses = [candidate]

        # Call the unbound method against our mock self
        EmbeddedScanner._post_scan_automation(es, result)

        # Row must be written
        assert out.exists(), "dry_run_validation.jsonl was not created"
        lines = [l for l in out.read_text().splitlines() if l.strip()]
        assert len(lines) >= 1, (
            "Expected at least one JSONL row for the directional candidate. "
            "If this test fails, the validation_stats wiring in "
            "embedded_scanner._post_scan_automation has regressed."
        )
        # Parse the row to ensure schema is valid
        row = json.loads(lines[0])
        assert row.get("pair") == "EUR_USD"

    def test_zero_candidates_writes_nothing(
        self,
        tmp_path: Path,
    ) -> None:
        """Empty analyses must not bloat the jsonl with null rows."""
        from src.scanner.automation.validation_stats import ScanDistributionStats
        from src.tui.embedded_scanner import EmbeddedScanner

        out = tmp_path / "dry_run_validation.jsonl"

        es = MagicMock()
        es._scan_count = 2
        es._brain = lambda *a, **kw: None
        es._scanner = MagicMock()
        es._scanner._observation_consumer = None
        es._online_rl = None
        es._config_adjuster = None
        es._config = None
        es._model_freshness = None
        es._validation_stats = ScanDistributionStats(log_path=out, enabled=True)

        result = MagicMock()
        result.analyses = []  # empty cycle

        EmbeddedScanner._post_scan_automation(es, result)

        # File may or may not exist; if it does, must be empty
        if out.exists():
            text = out.read_text()
            assert text.strip() == "", (
                "Empty-candidate cycle must not write rows. "
                "Got: " + text[:200]
            )
