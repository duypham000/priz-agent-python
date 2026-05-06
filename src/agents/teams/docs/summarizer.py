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
from src.agents.teams.docs.state import DocsTeamState

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM_PROMPT = (
    "You are a meeting documentation specialist. "
    "Given a meeting script, produce structured Meeting Minutes with sections: "
    "## Meeting Minutes, ## Key Decisions, ## Executive Summary, ## Discussion Points. "
    "Output ONLY the Meeting Minutes document."
)


def _load_prompt() -> str:
    path = Path(__file__).parents[4] / "prompts" / "teams" / "docs" / "summarizer.yaml"
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return data["system_prompt"]
    except Exception:
        logger.debug("summarizer prompt not found, using default")
        return _DEFAULT_SYSTEM_PROMPT


def make_meeting_minutes_node(
    model: BaseChatModel,
) -> Callable[[DocsTeamState], Coroutine[Any, Any, dict]]:
    """Return a LangGraph node that converts script → Meeting Minutes summary.

    Reads: script (fallback: raw_input)
    Writes: summary
    """
    system_prompt = _load_prompt()

    async def meeting_minutes_node(state: DocsTeamState) -> dict:
        content = state.get("script") or state.get("raw_input") or ""
        response = await model.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Script:\n\n{content}"),
        ])
        return {"summary": response.content}

    return meeting_minutes_node


class SummarizerAgent(BaseAgent):
    name = "summarizer"

    def build_graph(self) -> CompiledGraph:
        node = make_meeting_minutes_node(self.adapter.get_model())
        graph = StateGraph(DocsTeamState)
        graph.add_node("summarize", node)
        graph.add_edge(START, "summarize")
        graph.add_edge("summarize", END)
        return graph.compile()
