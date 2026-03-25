"""Aura ↔ Buddy Feedback Bridge.

Three signal types flow through the bridge:
1. Readiness Signal (Human → Domain): Aura's readiness score modulates Buddy's risk
2. Outcome Signal (Domain → Human): Buddy's trade outcomes feed Aura's pattern engine
3. Override Signal (Bidirectional): When the trader overrides Buddy, both systems learn
"""
