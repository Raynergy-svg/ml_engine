"""Scan diagnostics reporter — periodic signal quality report every 10 scan cycles.

Phase 56 US-350: Track signal quality across scans. Generate a periodic
diagnostic report every 10 scans showing confidence distribution, gate pass
rates, and a "stalled" flag when no trades have executed for 50+ scans.

The stalled flag is consumed by the Ralph PRD generator to surface
execution health issues for autonomous improvement analysis.

Usage:
    reporter = ScanDiagnosticsReporter()
    reporter.record_scan(pairs_results)
    if reporter.should_report():
        report = reporter.generate_report()
        reporter.save_report(report)
    reporter.save_state()
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_REPORT_PATH = _PROJECT_ROOT / "trained_data" / "scan_diagnostics_report.json"
_DEFAULT_STATE_PATH = _PROJECT_ROOT / "trained_data" / "scan_diagnostics_state.json"

# How many scans between reports
REPORT_INTERVAL = 10

# How many scans without a trade before "stalled" flag triggers
STALLED_THRESHOLD = 50

_STATE_VERSION = 1


@dataclass
class ScanRecord:
    """A single scan cycle result summary."""

    timestamp: str
    pairs_scanned: int
    tradeable_count: int
    top_confidence: float
    avg_confidence: float
    any_direction: bool  # Any pair generated a LONG/SHORT direction (not HOLD)


class ScanDiagnosticsReporter:
    """Track signal quality and generate periodic diagnostic reports.

    Accumulates scan results and generates a report every REPORT_INTERVAL
    scans showing: avg confidence, gate pass rate, top blocking patterns,
    and a stalled flag if no trades have generated in STALLED_THRESHOLD scans.

    Usage:
        reporter = ScanDiagnosticsReporter()
        reporter.record_scan(pairs_results_list)
        if reporter.should_report():
            report = reporter.generate_report()
            reporter.save_report(report)
        reporter.save_state()
    """

    def __init__(
        self,
        report_path: Optional[Path] = None,
        state_path: Optional[Path] = None,
        report_interval: int = REPORT_INTERVAL,
        stalled_threshold: int = STALLED_THRESHOLD,
    ):
        self.report_path = Path(report_path or _DEFAULT_REPORT_PATH)
        self.state_path = Path(state_path or _DEFAULT_STATE_PATH)
        self.report_interval = int(report_interval)
        self.stalled_threshold = int(stalled_threshold)

        # Persistent state
        self._total_scans: int = 0
        self._scans_since_last_report: int = 0
        self._scans_since_last_trade: int = 0
        self._total_trades_seen: int = 0

        # Rolling scan records (last REPORT_INTERVAL)
        self._recent_scans: List[ScanRecord] = []

        self._load_state()

    # ── Public API ──────────────────────────────────────────────────────────

    def record_scan(self, pairs_results: List[Dict[str, Any]]) -> None:
        """Record the results of one scan cycle.

        Args:
            pairs_results: List of pair result dicts. Each should have:
                - confidence (float): confidence score
                - is_tradeable (bool): whether the pair cleared all gates
                - direction (str): "LONG", "SHORT", or "HOLD"
        """
        if not pairs_results:
            return

        confs = [float(r.get("confidence", 0.0)) for r in pairs_results]
        tradeable = [r for r in pairs_results if r.get("is_tradeable", False)]
        tradeable_count = len(tradeable)
        any_direction = any(r.get("direction", "HOLD") in {"LONG", "SHORT"} for r in pairs_results)

        record = ScanRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            pairs_scanned=len(pairs_results),
            tradeable_count=tradeable_count,
            top_confidence=max(confs) if confs else 0.0,
            avg_confidence=sum(confs) / len(confs) if confs else 0.0,
            any_direction=any_direction,
        )

        self._recent_scans.append(record)
        # Keep only last REPORT_INTERVAL records
        if len(self._recent_scans) > self.report_interval:
            self._recent_scans = self._recent_scans[-self.report_interval:]

        self._total_scans += 1
        self._scans_since_last_report += 1

        if tradeable_count > 0:
            # A tradeable pair was found (trade may or may not have executed)
            self._scans_since_last_trade = 0
            self._total_trades_seen += tradeable_count
        else:
            self._scans_since_last_trade += 1

    def should_report(self) -> bool:
        """Return True every REPORT_INTERVAL scans."""
        return self._scans_since_last_report >= self.report_interval

    def generate_report(self) -> Dict[str, Any]:
        """Generate a diagnostic report for the current window of scans.

        Returns:
            Dict with:
                total_scans: Total scans since reporter initialized
                scans_since_last_report: Scans since last report
                scans_since_last_trade: Scans since a tradeable pair was found
                avg_top_score: Average top confidence across window
                avg_confidence: Average confidence across all pairs in window
                gate_pass_rate_pct: % of scans that found at least one tradeable pair
                pairs_with_direction_pct: % of scans where any pair had non-HOLD direction
                stalled: bool — True if scans_since_last_trade > stalled_threshold
                diagnostic_summary: str — human-readable diagnosis
                window_size: Number of scans in the current reporting window
                timestamp: Report generation timestamp
        """
        if not self._recent_scans:
            return {
                "total_scans": self._total_scans,
                "scans_since_last_report": self._scans_since_last_report,
                "scans_since_last_trade": self._scans_since_last_trade,
                "avg_top_score": 0.0,
                "avg_confidence": 0.0,
                "gate_pass_rate_pct": 0.0,
                "pairs_with_direction_pct": 0.0,
                "stalled": self._scans_since_last_trade >= self.stalled_threshold,
                "diagnostic_summary": "No scans recorded yet",
                "window_size": 0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        n = len(self._recent_scans)
        avg_top = sum(r.top_confidence for r in self._recent_scans) / n
        avg_conf = sum(r.avg_confidence for r in self._recent_scans) / n
        gate_pass_scans = sum(1 for r in self._recent_scans if r.tradeable_count > 0)
        direction_scans = sum(1 for r in self._recent_scans if r.any_direction)

        gate_pass_rate = round(gate_pass_scans / n * 100.0, 1)
        direction_pct = round(direction_scans / n * 100.0, 1)
        stalled = self._scans_since_last_trade >= self.stalled_threshold

        if stalled:
            diag = (
                f"STALLED: {self._scans_since_last_trade} consecutive scans with no tradeable pair. "
                f"Avg top confidence={avg_top:.3f}. Gate pass rate={gate_pass_rate}%. "
                f"Direction found in {direction_pct}% of scans. "
                "Likely cause: stacked confidence penalties suppressing all signals."
            )
        elif gate_pass_rate < 10.0:
            diag = (
                f"LOW_SIGNAL: Only {gate_pass_rate}% of scans found tradeable pairs. "
                f"Avg confidence={avg_conf:.3f}. Consider threshold review."
            )
        else:
            diag = (
                f"OK: Gate pass rate={gate_pass_rate}% over {n} scans. "
                f"Avg confidence={avg_conf:.3f}."
            )

        return {
            "total_scans": self._total_scans,
            "scans_since_last_report": self._scans_since_last_report,
            "scans_since_last_trade": self._scans_since_last_trade,
            "avg_top_score": round(avg_top, 4),
            "avg_confidence": round(avg_conf, 4),
            "gate_pass_rate_pct": gate_pass_rate,
            "pairs_with_direction_pct": direction_pct,
            "stalled": stalled,
            "diagnostic_summary": diag,
            "window_size": n,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def save_report(self, report: Dict[str, Any]) -> None:
        """Atomically write the report to the report path."""
        self._atomic_write(self.report_path, report)
        self._scans_since_last_report = 0  # Reset window counter

    # ── Persistence ─────────────────────────────────────────────────────────

    def save_state(self) -> None:
        """Atomically persist tracker state (scan counts) to disk."""
        state = {
            "version": _STATE_VERSION,
            "total_scans": self._total_scans,
            "scans_since_last_report": self._scans_since_last_report,
            "scans_since_last_trade": self._scans_since_last_trade,
            "total_trades_seen": self._total_trades_seen,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        self._atomic_write(self.state_path, state)

    def _load_state(self) -> None:
        """Load persisted state. Graceful fallback on any error."""
        if not self.state_path.exists():
            return
        try:
            raw = json.loads(self.state_path.read_text())
            if not isinstance(raw, dict):
                return
            self._total_scans = int(raw.get("total_scans", 0))
            self._scans_since_last_report = int(raw.get("scans_since_last_report", 0))
            self._scans_since_last_trade = int(raw.get("scans_since_last_trade", 0))
            self._total_trades_seen = int(raw.get("total_trades_seen", 0))
            logger.debug(
                "ScanDiagnosticsReporter: Loaded state (total_scans=%d, stalled_count=%d)",
                self._total_scans, self._scans_since_last_trade,
            )
        except Exception as e:
            logger.warning("ScanDiagnosticsReporter: Load state failed: %s", e)

    def _atomic_write(self, path: Path, data: Dict[str, Any]) -> None:
        """Atomically write JSON data to path."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(data, f, indent=2, sort_keys=True)
            os.rename(tmp_path, path)
        except Exception as e:
            logger.warning("ScanDiagnosticsReporter: Write to %s failed: %s", path, e)
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
