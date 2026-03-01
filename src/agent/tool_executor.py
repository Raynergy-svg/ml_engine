"""Sequential execution of planner tool calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .tool_registry import ToolRegistry
from .tool_schema import ToolCall, ToolResult


@dataclass
class ToolExecutor:
    registry: ToolRegistry
    state: dict[str, Any] = field(default_factory=dict)

    def execute_plan(self, steps: list[ToolCall]) -> list[ToolResult]:
        results: list[ToolResult] = []
        for step in steps:
            result = self.registry.execute(step.tool, step.args, self.state)
            results.append(result)
        return results
