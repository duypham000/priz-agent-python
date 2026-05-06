from __future__ import annotations

from src.core.exceptions import ToolError
from src.settings import settings

try:
    from tavily import AsyncTavilyClient
except ImportError:
    AsyncTavilyClient = None  # type: ignore[assignment,misc]


async def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Search the web via Tavily and return a list of results.

    Each result: {"title": str, "url": str, "content": str}
    Requires TAVILY_API_KEY in settings.
    """
    if not settings.tavily_api_key:
        raise ToolError(
            "Tavily API key is not configured (set TAVILY_API_KEY in .env)",
            tool_name="web_search",
            code="TAVILY_NOT_CONFIGURED",
        )
    if AsyncTavilyClient is None:
        raise ToolError(
            "tavily-python package is not installed",
            tool_name="web_search",
            code="TAVILY_NOT_INSTALLED",
        )

    try:
        client = AsyncTavilyClient(api_key=settings.tavily_api_key)
        response = await client.search(query, max_results=max_results)
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
            }
            for r in response.get("results", [])
        ]
    except Exception as exc:
        raise ToolError(
            f"Tavily search failed: {exc}",
            tool_name="web_search",
            code="TAVILY_SEARCH_ERROR",
        ) from exc
