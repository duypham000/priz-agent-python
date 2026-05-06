from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import Any, Callable, Coroutine

import yaml
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.pregel import Pregel as CompiledGraph

from src.agents.base import BaseAgent
from src.agents.teams.knowledge.state import KnowledgeTeamState

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM_PROMPT = (
    "You are a technical documentation analyst. Given raw documentation content, "
    "produce a concise Technical Markdown Summary that covers: key concepts, "
    "APIs/interfaces, usage patterns, important constraints, and code examples. "
    "Structure your output with clear headings. Be precise and developer-focused."
)

_MAX_DOC_CHARS = 8000


def _load_prompt() -> str:
    path = Path(__file__).parents[4] / "prompts" / "teams" / "knowledge" / "knowledge_scout.yaml"
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return data["system_prompt"]
    except Exception:
        logger.debug("knowledge_scout prompt not found, using default")
        return _DEFAULT_SYSTEM_PROMPT


def _fetch_source(source: str) -> str:
    """Fetch content from a URL or local file path. Returns plain text."""
    if source.startswith("http://") or source.startswith("https://"):
        try:
            with urllib.request.urlopen(source, timeout=10) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                return raw[:_MAX_DOC_CHARS]
        except Exception as exc:
            logger.warning("Failed to fetch %s: %s", source, exc)
            return f"[Could not fetch {source}: {exc}]"
    else:
        try:
            content = Path(source).read_text(encoding="utf-8", errors="replace")
            return content[:_MAX_DOC_CHARS]
        except Exception as exc:
            logger.warning("Failed to read %s: %s", source, exc)
            return f"[Could not read {source}: {exc}]"


def make_knowledge_scout_node(
    model: BaseChatModel,
) -> Callable[[KnowledgeTeamState], Coroutine[Any, Any, dict]]:
    """Return a LangGraph node that summarizes documentation sources.

    Reads: source_docs, messages (fallback)
    Writes: technical_summary
    """
    system_prompt = _load_prompt()

    async def knowledge_scout_node(state: KnowledgeTeamState) -> dict:
        source_docs = state.get("source_docs") or []

        if source_docs:
            parts = []
            for src in source_docs:
                content = _fetch_source(src)
                parts.append(f"### Source: {src}\n\n{content}")
            raw_input = "\n\n---\n\n".join(parts)
        else:
            messages = state.get("messages") or []
            last_content = messages[-1].content if messages else "No input provided."
            raw_input = str(last_content)

        response = await model.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=f"Analyze the following documentation and produce a Technical Markdown Summary:\n\n{raw_input}"
            ),
        ])
        return {"technical_summary": response.content}

    return knowledge_scout_node


class KnowledgeScoutAgent(BaseAgent):
    name = "knowledge_scout"

    def build_graph(self) -> CompiledGraph:
        node = make_knowledge_scout_node(self.adapter.get_model())
        graph = StateGraph(KnowledgeTeamState)
        graph.add_node("knowledge_scout", node)
        graph.add_edge(START, "knowledge_scout")
        graph.add_edge("knowledge_scout", END)
        return graph.compile()
