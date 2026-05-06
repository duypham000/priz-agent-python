from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Coroutine

import yaml
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.pregel import Pregel as CompiledGraph

from src.agents.base import BaseAgent
from src.agents.teams.research.state import ResearchTeamState
from src.core.exceptions import ToolError
from src.tools.builtin.web_search import web_search

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM_PROMPT = (
    "You are a market research specialist with access to real-time web search. "
    "Analyze search results thoroughly and provide comprehensive market intelligence. "
    "Structure your findings with clear sections: Key Findings, Market Trends, and Notable Sources."
)

_EXPANSION_PROMPT = (
    "Given this search result, generate ONE focused follow-up search query "
    "to find deeper, more specific information. "
    "Reply with ONLY the search query, nothing else.\n\n"
    "Original query: {query}\n"
    "Search result title: {title}\n"
    "Search result excerpt: {excerpt}"
)

_SYNTHESIS_PROMPT = (
    "You are a market research analyst. Synthesize the following web search results "
    "into a comprehensive market intelligence report. "
    "Include: Key Findings, Market Trends, Competitive Landscape, and Data Points.\n\n"
    "Original query: {query}\n\n"
    "Internal context (if any):\n{internal_context}\n\n"
    "Search results:\n{results}"
)


def _load_prompt() -> str:
    path = Path(__file__).parents[4] / "prompts" / "teams" / "research" / "market_scout.yaml"
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return data["system_prompt"]
    except Exception:
        logger.debug("market_scout prompt not found, using default")
        return _DEFAULT_SYSTEM_PROMPT


def _format_results(results: list[dict]) -> str:
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(f"[{i}] {r.get('title', 'Untitled')}\nURL: {r.get('url', '')}\n{r.get('content', '')[:400]}")
    return "\n\n".join(parts) if parts else "No results found."


def make_market_scout_node(
    model: BaseChatModel,
) -> Callable[[ResearchTeamState], Coroutine[Any, Any, dict]]:
    """Return a LangGraph node that performs LATS-style web search expansion.

    LATS (simplified 2-level tree):
      Level 0: Initial search with original query
      Evaluate: LLM picks top results and generates follow-up queries
      Level 1: Deeper searches on promising branches
      Synthesize: LLM merges all results

    Reads: query, internal_knowledge (optional context)
    Writes: web_results
    """
    synthesis_prompt = _load_prompt()

    async def market_scout_node(state: ResearchTeamState) -> dict:
        query = state.get("query") or ""
        internal_context = state.get("internal_knowledge") or "None"

        # Level 0: initial search
        all_results: list[dict] = []
        try:
            level0 = await web_search(query, max_results=5)
            all_results.extend(level0)
        except ToolError as exc:
            logger.warning("web_search level-0 failed: %s", exc)
            return {"web_results": f"Web search unavailable: {exc}"}

        # LATS expansion: pick top-2 results and generate follow-up queries
        promising = level0[:2]
        for result in promising:
            expansion_query = _EXPANSION_PROMPT.format(
                query=query,
                title=result.get("title", ""),
                excerpt=result.get("content", "")[:200],
            )
            try:
                followup_response = await model.ainvoke([HumanMessage(content=expansion_query)])
                followup_query = followup_response.content.strip()
                if followup_query:
                    level1 = await web_search(followup_query, max_results=3)
                    all_results.extend(level1)
            except (ToolError, Exception) as exc:
                logger.warning("LATS expansion failed: %s", exc)

        # Synthesize all results
        formatted = _format_results(all_results)
        response = await model.ainvoke([
            SystemMessage(content=synthesis_prompt),
            HumanMessage(
                content=_SYNTHESIS_PROMPT.format(
                    query=query,
                    internal_context=internal_context[:500],
                    results=formatted,
                )
            ),
        ])
        return {"web_results": response.content}

    return market_scout_node


class MarketScoutAgent(BaseAgent):
    name = "market_scout"

    def build_graph(self) -> CompiledGraph:
        node = make_market_scout_node(self.adapter.get_model())
        graph = StateGraph(ResearchTeamState)
        graph.add_node("market_scout", node)
        graph.add_edge(START, "market_scout")
        graph.add_edge("market_scout", END)
        return graph.compile()
