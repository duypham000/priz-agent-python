"""Unit tests for Phase 3 — Tool Registry."""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.exceptions import ToolError
from src.tools.registry import ToolRegistry, ToolSpec


# ─────────────────────────────────────────────────────────────────────────────
# TestToolRegistry
# ─────────────────────────────────────────────────────────────────────────────


class TestToolRegistry:
    class TestRegister:
        def test_register_validTool_storesSpec(self):
            # Arrange
            registry = ToolRegistry()

            async def my_tool(x: int) -> int:
                return x * 2

            # Act
            registry.register("double", my_tool, "doubles a number", {"x": "int"})

            # Assert
            assert "double" in registry.list_tools()
            spec = registry.get_spec("double")
            assert spec.name == "double"
            assert spec.func is my_tool
            assert spec.description == "doubles a number"

        def test_register_duplicateName_overwritesPrevious(self):
            # Arrange
            registry = ToolRegistry()

            async def v1() -> str:
                return "v1"

            async def v2() -> str:
                return "v2"

            registry.register("tool", v1, "version 1", {})

            # Act
            registry.register("tool", v2, "version 2", {})

            # Assert
            spec = registry.get_spec("tool")
            assert spec.func is v2
            assert spec.description == "version 2"

        def test_register_withPermissions_storesPermissions(self):
            # Arrange
            registry = ToolRegistry()

            async def secure_tool() -> None:
                pass

            # Act
            registry.register("secure", secure_tool, "desc", {}, permissions=["admin"])

            # Assert
            assert registry.get_spec("secure").permissions == ["admin"]

        def test_register_noPermissions_defaultsToEmpty(self):
            # Arrange
            registry = ToolRegistry()

            async def tool() -> None:
                pass

            # Act
            registry.register("t", tool, "desc", {})

            # Assert
            assert registry.get_spec("t").permissions == []

    class TestGetLangchainTools:
        def test_noTools_returnsEmptyList(self):
            # Arrange
            registry = ToolRegistry()

            # Act
            tools = registry.get_langchain_tools()

            # Assert
            assert tools == []

        def test_withAsyncTools_returnsBaseToolList(self):
            # Arrange
            registry = ToolRegistry()

            async def search(query: str) -> list:
                return []

            registry.register("search", search, "search the web", {"query": "str"})

            # Act
            tools = registry.get_langchain_tools()

            # Assert
            assert len(tools) == 1
            assert tools[0].name == "search"
            assert tools[0].description == "search the web"

        def test_withSyncTool_wrapsIntoAsync(self):
            # Arrange
            registry = ToolRegistry()

            def greet(name: str) -> str:
                return f"Hello {name}"

            registry.register("greet", greet, "greet user", {"name": "str"})

            # Act
            tools = registry.get_langchain_tools()

            # Assert
            assert len(tools) == 1
            assert tools[0].name == "greet"

        def test_multipleTools_returnsAll(self):
            # Arrange
            registry = ToolRegistry()
            for i in range(3):
                async def noop() -> None: pass
                registry.register(f"tool_{i}", noop, f"tool {i}", {})

            # Act
            tools = registry.get_langchain_tools()

            # Assert
            assert len(tools) == 3

    class TestExecute:
        async def test_knownTool_returnsResult(self):
            # Arrange
            registry = ToolRegistry()

            async def add(a: int, b: int) -> int:
                return a + b

            registry.register("add", add, "add two numbers", {})

            # Act
            result = await registry.execute("add", {"a": 3, "b": 4})

            # Assert
            assert result == 7

        async def test_syncTool_returnsResult(self):
            # Arrange
            registry = ToolRegistry()

            def multiply(x: int, y: int) -> int:
                return x * y

            registry.register("multiply", multiply, "multiply", {})

            # Act
            result = await registry.execute("multiply", {"x": 6, "y": 7})

            # Assert
            assert result == 42

        async def test_unknownTool_raisesToolError(self):
            # Arrange
            registry = ToolRegistry()

            # Act & Assert
            with pytest.raises(ToolError) as exc_info:
                await registry.execute("nonexistent", {})

            assert exc_info.value.tool_name == "nonexistent"
            assert exc_info.value.code == "TOOL_NOT_FOUND"

        async def test_toolRaisesException_wrapsInToolError(self):
            # Arrange
            registry = ToolRegistry()

            async def broken() -> None:
                raise ValueError("boom")

            registry.register("broken", broken, "broken tool", {})

            # Act & Assert
            with pytest.raises(ToolError) as exc_info:
                await registry.execute("broken", {})

            assert exc_info.value.tool_name == "broken"
            assert exc_info.value.code == "TOOL_EXECUTION_ERROR"
            assert "boom" in str(exc_info.value)

        async def test_toolRaisesToolError_propagatesUnwrapped(self):
            # Arrange
            registry = ToolRegistry()

            async def explicit_fail() -> None:
                raise ToolError("explicit", tool_name="explicit_fail", code="CUSTOM")

            registry.register("explicit_fail", explicit_fail, "desc", {})

            # Act & Assert
            with pytest.raises(ToolError) as exc_info:
                await registry.execute("explicit_fail", {})

            assert exc_info.value.code == "CUSTOM"

    class TestListTools:
        def test_returnsRegisteredNames(self):
            # Arrange
            registry = ToolRegistry()

            async def t1() -> None: pass
            async def t2() -> None: pass

            registry.register("alpha", t1, "a", {})
            registry.register("beta", t2, "b", {})

            # Act
            names = registry.list_tools()

            # Assert
            assert set(names) == {"alpha", "beta"}

        def test_emptyRegistry_returnsEmptyList(self):
            # Arrange
            registry = ToolRegistry()

            # Act & Assert
            assert registry.list_tools() == []

    class TestGetSpec:
        def test_existingTool_returnsSpec(self):
            # Arrange
            registry = ToolRegistry()

            async def noop() -> None: pass

            registry.register("noop", noop, "does nothing", {})

            # Act
            spec = registry.get_spec("noop")

            # Assert
            assert isinstance(spec, ToolSpec)
            assert spec.name == "noop"

        def test_missingTool_raisesToolError(self):
            # Arrange
            registry = ToolRegistry()

            # Act & Assert
            with pytest.raises(ToolError) as exc_info:
                registry.get_spec("missing")

            assert exc_info.value.code == "TOOL_NOT_FOUND"


# ─────────────────────────────────────────────────────────────────────────────
# TestBuiltinGuidelines
# ─────────────────────────────────────────────────────────────────────────────


class TestBuiltinGuidelines:
    def test_fileExists_returnsContent(self, tmp_path: Path):
        # Arrange
        from src.tools.builtin import guidelines as mod

        agent_dir = tmp_path / "prompts" / "manager"
        agent_dir.mkdir(parents=True)
        (agent_dir / "system.yaml").write_text("role: manager\n", encoding="utf-8")

        original_root = mod._PROJECT_ROOT
        mod._PROJECT_ROOT = tmp_path  # patch project root

        # Act
        try:
            content = mod.read_guidelines("manager")
        finally:
            mod._PROJECT_ROOT = original_root

        # Assert
        assert content == "role: manager\n"

    def test_fileMissing_raisesToolError(self, tmp_path: Path):
        # Arrange
        from src.tools.builtin import guidelines as mod

        original_root = mod._PROJECT_ROOT
        mod._PROJECT_ROOT = tmp_path

        # Act & Assert
        try:
            with pytest.raises(ToolError) as exc_info:
                mod.read_guidelines("nonexistent_agent")
        finally:
            mod._PROJECT_ROOT = original_root

        assert exc_info.value.code == "GUIDELINES_NOT_FOUND"
        assert exc_info.value.tool_name == "read_guidelines"


# ─────────────────────────────────────────────────────────────────────────────
# TestBuiltinWebSearch
# ─────────────────────────────────────────────────────────────────────────────


class TestBuiltinWebSearch:
    async def test_apiKeyMissing_raisesToolError(self):
        # Arrange
        from src.tools.builtin import web_search as mod

        with patch.object(mod.settings, "tavily_api_key", ""):
            # Act & Assert
            with pytest.raises(ToolError) as exc_info:
                await mod.web_search("test query")

        assert exc_info.value.code == "TAVILY_NOT_CONFIGURED"

    async def test_withMockClient_returnsResults(self):
        # Arrange
        from src.tools.builtin import web_search as mod

        mock_response = {
            "results": [
                {"title": "Result 1", "url": "https://example.com", "content": "body 1"},
                {"title": "Result 2", "url": "https://example.org", "content": "body 2"},
            ]
        }
        mock_client = MagicMock()
        mock_client.search = AsyncMock(return_value=mock_response)

        with (
            patch.object(mod.settings, "tavily_api_key", "fake-key"),
            patch("src.tools.builtin.web_search.AsyncTavilyClient", return_value=mock_client),
        ):
            # Act
            results = await mod.web_search("AI news", max_results=2)

        # Assert
        assert len(results) == 2
        assert results[0]["title"] == "Result 1"
        assert results[1]["url"] == "https://example.org"

    async def test_tavilyError_raisesToolError(self):
        # Arrange
        from src.tools.builtin import web_search as mod

        mock_client = MagicMock()
        mock_client.search = AsyncMock(side_effect=RuntimeError("API down"))

        with (
            patch.object(mod.settings, "tavily_api_key", "fake-key"),
            patch("src.tools.builtin.web_search.AsyncTavilyClient", return_value=mock_client),
        ):
            # Act & Assert
            with pytest.raises(ToolError) as exc_info:
                await mod.web_search("query")

        assert exc_info.value.code == "TAVILY_SEARCH_ERROR"


# ─────────────────────────────────────────────────────────────────────────────
# TestBuiltinFileReader
# ─────────────────────────────────────────────────────────────────────────────


class TestBuiltinFileReader:
    async def test_localFile_returnsContent(self, tmp_path: Path):
        # Arrange
        from src.tools.builtin.file_reader import read_file

        target = tmp_path / "sample.txt"
        target.write_text("hello world", encoding="utf-8")

        # Act
        content = await read_file(str(target))

        # Assert
        assert content == "hello world"

    async def test_localFileMissing_raisesToolError(self):
        # Arrange
        from src.tools.builtin.file_reader import read_file

        # Act & Assert
        with pytest.raises(ToolError) as exc_info:
            await read_file("/nonexistent/path/file.txt")

        assert exc_info.value.code == "FILE_NOT_FOUND"

    async def test_httpUrl_returnsContent(self):
        # Arrange
        from src.tools.builtin.file_reader import read_file

        mock_response = MagicMock()
        mock_response.text = "<html>content</html>"
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch("src.tools.builtin.file_reader.httpx.AsyncClient", return_value=mock_client):
            # Act
            content = await read_file("https://example.com/page")

        # Assert
        assert content == "<html>content</html>"

    async def test_httpTimeout_raisesToolError(self):
        # Arrange
        import httpx
        from src.tools.builtin.file_reader import read_file

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("timed out"))

        with patch("src.tools.builtin.file_reader.httpx.AsyncClient", return_value=mock_client):
            # Act & Assert
            with pytest.raises(ToolError) as exc_info:
                await read_file("https://slow.example.com/")

        assert exc_info.value.code == "HTTP_TIMEOUT"

    async def test_httpStatusError_raisesToolError(self):
        # Arrange
        import httpx
        from src.tools.builtin.file_reader import read_file

        fake_response = MagicMock()
        fake_response.status_code = 404

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(
            side_effect=httpx.HTTPStatusError("not found", request=MagicMock(), response=fake_response)
        )

        with patch("src.tools.builtin.file_reader.httpx.AsyncClient", return_value=mock_client):
            # Act & Assert
            with pytest.raises(ToolError) as exc_info:
                await read_file("https://example.com/missing")

        assert exc_info.value.code == "HTTP_ERROR"


# ─────────────────────────────────────────────────────────────────────────────
# TestBuiltinCalendar
# ─────────────────────────────────────────────────────────────────────────────


class TestBuiltinCalendar:
    async def test_createTask_logBackend_returnsLoggedResponse(self):
        # Arrange — default backend is "log"
        from src.tools.builtin.calendar import create_task

        # Act
        result = await create_task("Review PR", "2025-05-01", "alice")

        # Assert
        assert result["status"] == "logged"
        assert result["title"] == "Review PR"
        assert result["deadline"] == "2025-05-01"
        assert result["owner"] == "alice"

    async def test_listTasks_returnsEmptyList(self):
        # Arrange
        from src.tools.builtin.calendar import list_tasks

        # Act
        result = await list_tasks()

        # Assert
        assert result == []


# ─────────────────────────────────────────────────────────────────────────────
# TestBuiltinCodeRunner
# ─────────────────────────────────────────────────────────────────────────────


class TestBuiltinCodeRunner:
    async def test_validCode_returnsStdout(self):
        # Arrange
        from src.tools.builtin.code_runner import run_code

        # Act
        result = await run_code("print('hello')")

        # Assert
        assert result["returncode"] == 0
        assert "hello" in result["stdout"]
        assert result["stderr"] == ""

    async def test_syntaxError_returnsStderr(self):
        # Arrange
        from src.tools.builtin.code_runner import run_code

        # Act
        result = await run_code("def bad(: pass")

        # Assert
        assert result["returncode"] != 0
        assert result["stderr"] != ""

    async def test_runtimeError_returnsNonZeroReturncode(self):
        # Arrange
        from src.tools.builtin.code_runner import run_code

        # Act
        result = await run_code("raise ValueError('oops')")

        # Assert
        assert result["returncode"] != 0
        assert "ValueError" in result["stderr"]

    async def test_timeout_raisesToolError(self):
        # Arrange
        from src.tools.builtin.code_runner import run_code

        # Act & Assert
        with pytest.raises(ToolError) as exc_info:
            await run_code("import time; time.sleep(60)", timeout=1)

        assert exc_info.value.code == "TIMEOUT"
        assert exc_info.value.tool_name == "run_code"

    async def test_outputCapture_stdoutAndStderr(self):
        # Arrange
        from src.tools.builtin.code_runner import run_code

        code = "import sys; print('out'); print('err', file=sys.stderr)"

        # Act
        result = await run_code(code)

        # Assert
        assert "out" in result["stdout"]
        assert "err" in result["stderr"]


# ─────────────────────────────────────────────────────────────────────────────
# TestMCPClient
# ─────────────────────────────────────────────────────────────────────────────


class TestMCPClient:
    def test_notConnected_isConnectedReturnsFalse(self):
        # Arrange
        from src.tools.mcp.client import MCPClient

        registry = ToolRegistry()
        client = MCPClient(registry)

        # Act & Assert
        assert client.is_connected() is False

    async def test_disconnect_setsConnectedFalse(self):
        # Arrange
        from src.tools.mcp.client import MCPClient

        registry = ToolRegistry()
        client = MCPClient(registry)
        client._connected = True

        # Act
        await client.disconnect()

        # Assert
        assert client.is_connected() is False

    async def test_connect_mcpImportError_gracefullyHandled(self):
        # Arrange
        from src.tools.mcp.client import MCPClient

        registry = ToolRegistry()
        client = MCPClient(registry)

        with patch.dict("sys.modules", {"mcp": None, "mcp.client.sse": None}):
            with patch("builtins.__import__", side_effect=ImportError("no mcp")):
                # Act
                await client.connect("http://localhost:3000/sse")

        # Assert — should not raise, stays disconnected
        assert client.is_connected() is False

    async def test_connect_registersToolsInRegistry(self):
        # Arrange
        from src.tools.mcp.client import MCPClient

        registry = ToolRegistry()
        client = MCPClient(registry)

        # Mock tool definition
        mock_tool = MagicMock()
        mock_tool.name = "mcp_search"
        mock_tool.description = "MCP search tool"
        mock_tool.inputSchema = {"query": "str"}

        mock_tools_result = MagicMock()
        mock_tools_result.tools = [mock_tool]

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=mock_tools_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_sse_stream = AsyncMock()
        mock_sse_stream.__aenter__ = AsyncMock(return_value=(AsyncMock(), AsyncMock()))
        mock_sse_stream.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("src.tools.mcp.client.sse_client", return_value=mock_sse_stream),
            patch("src.tools.mcp.client.ClientSession", return_value=mock_session),
        ):
            # Act
            await client.connect("http://localhost:3000/sse")

        # Assert
        assert "mcp_search" in registry.list_tools()
