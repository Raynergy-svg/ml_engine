"""Aura — Human Intelligence Engine.

The human-side recursive intelligence layer for the Buddy+Aura system.
Tracks emotional state, cognitive patterns, decision history, and
computes readiness signals that modulate Buddy's trading behavior.

Architecture:
    Conversations → Self-Model Graph → Pattern Engine → Readiness Score
        ↑                                                    ↓
        └── Insight Delivery ← Cross-Domain Correlations ← Buddy Outcomes
"""
