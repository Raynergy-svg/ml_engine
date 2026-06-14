#!/usr/bin/env python3
"""Historical shadow validation for LLM Macro Agent (US-006).

Runs the LLM Macro Agent against known historical macro events to measure
whether it would have correctly signaled a regime shift before the event.

Events tested:
  - July 2024 BoJ hike ( surprise rate increase )
  - September 2024 Fed cut ( first cut in cycle )
  - March 2023 SVB collapse / Fed emergency lending
  - February 2022 Russia-Ukraine invasion (safe-haven shock)

Usage:
    python scripts/validate_llm_macro_shadow.py

Output:
    trained_data/reports/llm_macro_shadow_validation.md
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scanner.agents.llm_macro_agent import LLMMacroAgent, LLMMacroAgentConfig
from src.intelligence.cb_audio_processor import CBAudioProcessor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ------------------------------------------------------------------
# Historical event fixtures
# ------------------------------------------------------------------

@dataclass
class HistoricalEvent:
    name: str
    date: str
    pair: str
    direction: str
    pre_event_transcript: str
    expected_regime_shift: float = 0.7  # threshold we want agent to exceed


HISTORICAL_EVENTS: List[HistoricalEvent] = [
    HistoricalEvent(
        name="July 2024 BoJ Hike",
        date="2024-07-31",
        pair="USD_JPY",
        direction="SHORT",
        pre_event_transcript=(
            "Recent wage data shows sustained increases above the BoJ's expectations. "
            "Core inflation has remained above the 2% target for 27 consecutive months. "
            "Governor Ueda has signaled that accommodative conditions are no longer necessary. "
            "The weak yen is hurting import costs and household purchasing power. "
            "Markets are pricing a 60% probability of a rate hike at the upcoming meeting."
        ),
        expected_regime_shift=0.7,
    ),
    HistoricalEvent(
        name="September 2024 Fed Cut",
        date="2024-09-18",
        pair="EUR_USD",
        direction="LONG",
        pre_event_transcript=(
            "The labor market is showing signs of cooling with nonfarm payrolls revisions to the downside. "
            "Core PCE inflation has decelerated to 2.5% annualized. "
            "Several FOMC members have hinted at a recalibration of policy. "
            "The dot plot in June suggested two cuts this year. "
            "Chair Powell emphasized data-dependence at Jackson Hole."
        ),
        expected_regime_shift=0.7,
    ),
    HistoricalEvent(
        name="March 2023 SVB Collapse",
        date="2023-03-10",
        pair="USD_JPY",
        direction="SHORT",
        pre_event_transcript=(
            "Regional bank stress is emerging with deposit outflows accelerating. "
            "The Fed has expanded its discount window and created a new Bank Term Funding Program. "
            "Treasury Secretary Yellen convened emergency meetings over the weekend. "
            "Market-implied rate path has collapsed by over 100bps in 48 hours. "
            "Flight-to-quality flows are benefitting Treasuries and the yen."
        ),
        expected_regime_shift=0.8,
    ),
    HistoricalEvent(
        name="February 2022 Russia-Ukraine Invasion",
        date="2022-02-24",
        pair="USD_CHF",
        direction="LONG",
        pre_event_transcript=(
            "Troops have massed at the Ukrainian border for weeks with satellite imagery confirming movements. "
            "Western leaders have warned of severe sanctions including SWIFT exclusion. "
            "Oil prices have climbed above $90/barrel on supply disruption fears. "
            "Safe-haven currencies are seeing inflows ahead of the weekend. "
            "Geopolitical risk indicators are at their highest since 2014."
        ),
        expected_regime_shift=0.8,
    ),
]


# ------------------------------------------------------------------
# Validation runner
# ------------------------------------------------------------------

@dataclass
class ValidationResult:
    event_name: str
    event_date: str
    pair: str
    regime_shift_probability: float
    direction_bias: str
    confidence: float
    passed_threshold: bool
    key_drivers: List[str]
    tone_score: Optional[float]
    notes: str = ""


def run_validation() -> List[ValidationResult]:
    """Run LLM Macro Agent against each historical event."""
    agent = LLMMacroAgent(LLMMacroAgentConfig(shadow_mode=True, enable_fred=False))
    processor = CBAudioProcessor()
    results: List[ValidationResult] = []

    for event in HISTORICAL_EVENTS:
        logger.info("Validating: %s (%s)", event.name, event.date)

        # Simulate agent evaluation using pre-event transcript
        # We use the processor's text analysis since we don't have audio
        tone = processor.process(speaker="system", transcript_text=event.pre_event_transcript)

        # Build a mock context for the LLM macro agent
        # In shadow/degraded mode, the agent will fall back to heuristic reasoning
        # based on the macro context it builds internally
        macro_context = agent._build_macro_context(event.pair)
        macro_context["historical_event"] = event.name
        macro_context["event_date"] = event.date
        macro_context["transcript_summary"] = event.pre_event_transcript[:500]

        # Query the LLM (degraded mode if no API keys)
        reasoning = agent._query_llm(macro_context, event.pair, event.direction)

        if reasoning is None:
            # Degraded mode: estimate from text analysis
            hawkish = sum(1 for w in ["hike", "tighten", "raise", "restrictive", "elevated"] if w in event.pre_event_transcript.lower())
            dovish = sum(1 for w in ["cut", "ease", "pause", "patient", "data-dependent"] if w in event.pre_event_transcript.lower())
            total = hawkish + dovish
            if total > 0:
                prob = 0.5 + 0.3 * ((hawkish - dovish) / total)
            else:
                prob = 0.5
            prob = max(0.0, min(1.0, prob))
            results.append(ValidationResult(
                event_name=event.name,
                event_date=event.date,
                pair=event.pair,
                regime_shift_probability=prob,
                direction_bias=event.direction,
                confidence=0.5,
                passed_threshold=prob >= event.expected_regime_shift,
                key_drivers=["degraded_mode_heuristic"],
                tone_score=tone.dovish_hawkish_score if not tone.degraded else None,
                notes="LLM unavailable; heuristic fallback used",
            ))
        else:
            results.append(ValidationResult(
                event_name=event.name,
                event_date=event.date,
                pair=event.pair,
                regime_shift_probability=reasoning.regime_shift_probability,
                direction_bias=reasoning.direction_bias,
                confidence=reasoning.confidence,
                passed_threshold=reasoning.regime_shift_probability >= event.expected_regime_shift,
                key_drivers=reasoning.key_drivers,
                tone_score=tone.dovish_hawkish_score if not tone.degraded else None,
                notes="LLM reasoning available",
            ))

    return results


def generate_report(results: List[ValidationResult]) -> str:
    """Generate markdown report."""
    lines = [
        "# LLM Macro Agent — Historical Shadow Validation Report",
        "",
        f"**Generated:** {datetime.now().isoformat()}",
        "**Method:** Pre-event transcripts fed to LLM Macro Agent + CB Audio Processor (tone analysis)",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Events tested | {len(results)} |",
        f"| Threshold passes | {sum(1 for r in results if r.passed_threshold)} / {len(results)} |",
        f"| Avg regime_shift_probability | {sum(r.regime_shift_probability for r in results) / len(results):.3f} |",
        "",
        "## Results by Event",
        "",
    ]

    for r in results:
        status = "✅ PASS" if r.passed_threshold else "❌ FAIL"
        lines.extend([
            f"### {r.event_name} ({r.event_date})",
            "",
            f"- **Pair:** {r.pair}",
            f"- **Direction bias:** {r.direction_bias}",
            f"- **Regime shift probability:** {r.regime_shift_probability:.3f}",
            f"- **Threshold:** {r.passed_threshold}",
            f"- **Confidence:** {r.confidence:.3f}",
            f"- **Tone score:** {r.tone_score:.3f}" if r.tone_score is not None else "- **Tone score:** N/A",
            f"- **Key drivers:** {', '.join(r.key_drivers)}",
            f"- **Notes:** {r.notes}",
            f"- **Status:** {status}",
            "",
        ])

    lines.extend([
        "## Conclusion",
        "",
        "This report validates whether the LLM Macro Agent (in degraded or full mode) "
        "would have flagged regime shifts before known historical macro events. "
        "A passing score means the agent's regime_shift_probability exceeded the "
        "expected threshold for that event.",
        "",
        "**Next steps:**",
        "- If >=3/4 events pass: consider promoting LLM Macro Agent from shadow to active voting.",
        "- If <2/4 events pass: investigate why the agent misses regime shifts (macro context gaps, threshold tuning).",
        "",
    ])

    return "\n".join(lines)


def main() -> int:
    """Entry point."""
    logger.info("Starting LLM Macro Agent historical shadow validation...")
    results = run_validation()
    report_md = generate_report(results)

    report_path = Path("trained_data/reports/llm_macro_shadow_validation.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_md, encoding="utf-8")

    # Also save JSON for programmatic consumption
    json_path = Path("trained_data/reports/llm_macro_shadow_validation.json")
    json_path.write_text(
        json.dumps([r.__dict__ for r in results], indent=2, default=str),
        encoding="utf-8",
    )

    logger.info("Report written to %s", report_path)
    logger.info("JSON data written to %s", json_path)
    logger.info("Pass rate: %d/%d", sum(1 for r in results if r.passed_threshold), len(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
