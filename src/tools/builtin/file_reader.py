from __future__ import annotations

from pathlib import Path

import httpx

from src.core.exceptions import ToolError

_HTTP_TIMEOUT = 30.0


async def read_file(path_or_url: str) -> str:
    """Read a local file or HTTP URL and return its plain text content."""
    if path_or_url.startswith(("http://", "https://")):
        return await _read_url(path_or_url)
    return _read_local(path_or_url)


def _read_local(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ToolError(
            f"File not found: {path}",
            tool_name="read_file",
            code="FILE_NOT_FOUND",
        ) from exc
    except OSError as exc:
        raise ToolError(
            f"Failed to read file '{path}': {exc}",
            tool_name="read_file",
            code="FILE_READ_ERROR",
        ) from exc


async def _read_url(url: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.text
    except httpx.TimeoutException as exc:
        raise ToolError(
            f"Request timed out for URL: {url}",
            tool_name="read_file",
            code="HTTP_TIMEOUT",
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise ToolError(
            f"HTTP {exc.response.status_code} for URL: {url}",
            tool_name="read_file",
            code="HTTP_ERROR",
        ) from exc
    except Exception as exc:
        raise ToolError(
            f"Failed to fetch URL '{url}': {exc}",
            tool_name="read_file",
            code="HTTP_FETCH_ERROR",
        ) from exc
