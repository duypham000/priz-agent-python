from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable

from langchain_core.tools import BaseTool, StructuredTool

from src.core.exceptions import ToolError



@dataclass
class ToolSpec:
    name: str
    func: Callable[..., Any]
    description: str
    schema: dict
    permissions: list[str] = field(default_factory=list)


class ToolRegistry:
    """3-layer tool registry.

    Layer 1 — register():         store tool spec in memory
    Layer 2 — get_langchain_tools(): expose as list[BaseTool] for LLM binding
    Layer 3 — execute():          async invocation with unified error handling
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    # ── Layer 1 ──────────────────────────────────────────────────────────────

    def register(
        self,
        name: str,
        func: Callable[..., Any],
        description: str,
        schema: dict,
        permissions: list[str] | None = None,
    ) -> None:
        self._tools[name] = ToolSpec(
            name=name,
            func=func,
            description=description,
            schema=schema,
            permissions=permissions or [],
        )

    # ── Layer 2 ──────────────────────────────────────────────────────────────

    def get_langchain_tools(self, agent_name: str | None = None) -> list[BaseTool]:
        """Convert registered tools to LangChain BaseTool list filtered by agent permissions."""
        tools: list[BaseTool] = []
        for spec in self._tools.values():
            if spec.permissions and agent_name not in spec.permissions:
                continue
            if inspect.iscoroutinefunction(spec.func):
                tool = StructuredTool.from_function(
                    coroutine=spec.func,
                    name=spec.name,
                    description=spec.description,
                )
            else:
                tool = StructuredTool.from_function(
                    func=spec.func,
                    name=spec.name,
                    description=spec.description,
                )
            tools.append(tool)
        return tools

    # ── Layer 3 ──────────────────────────────────────────────────────────────

    async def execute(self, name: str, args: dict[str, Any], agent_name: str | None = None) -> Any:
        if name not in self._tools:
            raise ToolError(
                f"Tool '{name}' is not registered",
                tool_name=name,
                code="TOOL_NOT_FOUND",
            )
        spec = self._tools[name]
        if spec.permissions and agent_name not in spec.permissions:
            raise ToolError(
                f"Agent '{agent_name}' does not have permission to use tool '{name}'",
                tool_name=name,
                code="TOOL_PERMISSION_DENIED",
            )
        try:
            if inspect.iscoroutinefunction(spec.func):
                return await spec.func(**args)
            return spec.func(**args)
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(
                f"Tool '{name}' raised an error: {exc}",
                tool_name=name,
                code="TOOL_EXECUTION_ERROR",
            ) from exc

    # ── Helpers ───────────────────────────────────────────────────────────────

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def get_spec(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise ToolError(
                f"Tool '{name}' is not registered",
                tool_name=name,
                code="TOOL_NOT_FOUND",
            )
        return self._tools[name]
