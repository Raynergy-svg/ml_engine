from __future__ import annotations

from src.agent.tool_executor import ToolExecutor
from src.agent.tool_registry import ToolRegistry
from src.agent.tool_schema import ToolCall, ToolResult, ToolSpec


def test_tool_executor_runs_registered_tools_in_order():
    registry = ToolRegistry()

    registry.register(
        ToolSpec(name="first", description="first tool"),
        lambda args, state: ToolResult(ok=True, tool_name="first", args=args, result={"seen": state.get("value")}, human_summary="first ok"),
    )
    registry.register(
        ToolSpec(name="second", description="second tool"),
        lambda args, state: ToolResult(ok=True, tool_name="second", args=args, result={"done": True}, human_summary="second ok"),
    )

    executor = ToolExecutor(registry, state={"value": 7})
    results = executor.execute_plan([ToolCall("first"), ToolCall("second", {"x": 1})])

    assert [result.tool_name for result in results] == ["first", "second"]
    assert results[0].result["seen"] == 7
    assert results[1].args == {"x": 1}
