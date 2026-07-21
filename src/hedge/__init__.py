"""AXIOM Hedge Layer — SHADOW / ANALYSIS-ONLY exposure decomposition + netting.

No execution path lives here. Nothing in this package places, modifies, or
cancels an order; unhalts a lane; arms a live gate; bypasses a gate; or
changes leverage. It reads position-shaped data passed in by the caller and
returns exposure reports. See ``docs/`` for the phase roadmap (Phase 1
tagging + Phase 2 netting are foundation; decision/execution is Phase 3+,
not implemented here).
"""
