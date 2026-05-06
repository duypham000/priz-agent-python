from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from src.core.exceptions import ToolError

if TYPE_CHECKING:
    from src.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

try:
    from mcp import ClientSession
    from mcp.client.sse import sse_client
    _MCP_AVAILABLE = True
except ImportError:
    ClientSession = None  # type: ignore[assignment,misc]
    sse_client = None  # type: ignore[assignment]
    _MCP_AVAILABLE = False


class MCPClient:
    """Bridge between an MCP server and ToolRegistry.

    Connects to an MCP server, discovers its tools, and registers each
    as an async callable in the provided ToolRegistry.
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry
        self._connected = False
        self._url: str | None = None
        self._session: Any = None

    async def connect(self, url: str) -> None:
        """Connect to MCP server and register discovered tools."""
        if not _MCP_AVAILABLE:
            logger.warning("mcp package not installed — MCP bridge disabled")
            return

        self._url = url
        try:
            async with sse_client(url) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._session = session
                    self._connected = True
                    result = await session.list_tools()
                    for tool in result.tools:
                        self._register_mcp_tool(tool, session)
                    logger.info(
                        "MCP bridge connected to %s — registered %d tools",
                        url,
                        len(result.tools),
                    )
        except Exception as exc:
            logger.warning("MCP connect failed for %s: %s", url, exc)
            self._connected = False

    def _register_mcp_tool(self, tool: Any, session: Any) -> None:
        tool_name: str = tool.name
        description: str = tool.description or ""
        input_schema: dict = tool.inputSchema or {}

        async def _call(**kwargs: Any) -> Any:
            try:
                result = await session.call_tool(tool_name, kwargs)
                return result.content
            except Exception as exc:
                raise ToolError(
                    f"MCP tool '{tool_name}' failed: {exc}",
                    tool_name=tool_name,
                    code="MCP_TOOL_ERROR",
                ) from exc

        self._registry.register(
            name=tool_name,
            func=_call,
            description=description,
            schema=input_schema,
            permissions=["mcp"],
        )

    async def disconnect(self) -> None:
        self._connected = False
        self._session = None

    def is_connected(self) -> bool:
        return self._connected
