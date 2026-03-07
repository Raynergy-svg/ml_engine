"""Sequential execution of planner tool calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .tool_registry import ToolRegistry
from .tool_schema import ToolCall, ToolResult


@dataclass
class ToolExecutor:
    registry: ToolRegistry
    state: dict[str, Any] = field(default_factory=dict)

    def execute_plan(
        self,
        steps: list[ToolCall],
        *,
        progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> list[ToolResult]:
        results: list[ToolResult] = []
        total_steps = len(steps)
        for index, step in enumerate(steps, start=1):
            if progress_callback is not None:
                progress_callback(
                    "tool_start",
                    {
                        "index": index,
                        "total": total_steps,
                        "tool": step.tool,
                        "args": dict(step.args or {}),
                    },
                )
            result = self.registry.execute(step.tool, step.args, self.state)
            results.append(result)
            if progress_callback is not None:
                progress_callback(
                    "tool_done",
                    {
                        "index": index,
                        "total": total_steps,
                        "tool": step.tool,
                        "ok": bool(result.ok),
                        "error": result.error,
                    },
                )
        return results
