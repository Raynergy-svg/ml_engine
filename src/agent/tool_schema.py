"""Structured tool schema for Buddy's planner runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class ToolArgSpec:
    name: str
    type_name: str
    description: str
    required: bool = False


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args: tuple[ToolArgSpec, ...] = ()

    def prompt_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "args_schema": {
                arg.name: {
                    "type": arg.type_name,
                    "description": arg.description,
                    "required": arg.required,
                }
                for arg in self.args
            },
        }


@dataclass
class ToolCall:
    tool: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    ok: bool
    tool_name: str
    args: dict[str, Any]
    result: dict[str, Any] = field(default_factory=dict)
    human_summary: str = ""
    error: str | None = None


ToolHandler = Callable[[dict[str, Any], dict[str, Any]], ToolResult]
