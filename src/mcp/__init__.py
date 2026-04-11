"""Buddy MCP package.

Exposes the autonomous ML trading system (scanner, feedback loop,
diagnostics, self-heal, trade state) over the Model Context Protocol so a
Claude orchestrator can observe and trigger the system via stdio.

The concrete server lives in :mod:`src.mcp.buddy_server`. This file is an
intentionally empty package marker — no heavy imports at package load time.
"""
