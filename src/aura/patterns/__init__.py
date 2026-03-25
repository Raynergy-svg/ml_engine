"""Aura pattern engine — multi-tier behavioral pattern detection.

T1 (daily):   Frequency-based patterns — emotional, override, readiness
T2 (weekly):  Cross-domain correlations — human state × trading outcome
T3 (monthly): Narrative arcs — long-term behavioral trajectories
Cloud:        Optional LLM synthesis for deeper pattern explanations

Usage:
    from src.aura.patterns.engine import PatternEngine

    engine = PatternEngine()
    results = engine.run_all(conversations, readiness_history)
    arcs = engine.get_narrative_arcs()
    report = engine.format_patterns_report()
"""

# Lazy imports to avoid heavy dependency loading in __init__
# (per .claude/rules/improvement.md code quality gates)
