"""Sonnet brain loop — research orchestration around a validated execution engine.

Hard boundary (see docs/superpowers/specs/2026-07-01-sonnet-brain-loop-design.md):
this package NEVER places a trade and NEVER arms/bypasses the human checkpoint for
flatten / set_gross_leverage / start_loop / unhalt. It proposes; existing gates and
the operator decide. De-risk (halt, bounded risk tightening) is the one class of
live-affecting action it may take autonomously, mirroring the precedented
autonomous set_halted(True) circuit breakers already in src/scanner/execution.py
and src/scanner/engine.py.
"""
