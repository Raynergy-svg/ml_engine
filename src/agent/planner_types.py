"""Planner runtime types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .tool_schema import ToolCall, ToolResult


@dataclass
class PlannerPlan:
    needs_clarification: bool = False
    clarification_question: str | None = None
    steps: list[ToolCall] = field(default_factory=list)
    final_answer: str | None = None


@dataclass
class PlannerRuntimeResponse:
    plan: PlannerPlan
    tool_results: list[ToolResult] = field(default_factory=list)
    answer: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
