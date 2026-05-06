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
    "You are a data processing specialist. "
    "Convert the raw input into a clean, structured Markdown script. "
    "Add [HH:MM] timestamps if detectable. "
    "Output ONLY the Markdown script."
)


def _load_prompt() -> str:
    path = Path(__file__).parents[4] / "prompts" / "teams" / "docs" / "data_processor.yaml"
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return data["system_prompt"]
    except Exception:
        logger.debug("data_processor prompt not found, using default")
        return _DEFAULT_SYSTEM_PROMPT


def make_data_processor_node(
    model: BaseChatModel,
) -> Callable[[DocsTeamState], Coroutine[Any, Any, dict]]:
    """Return a LangGraph node that converts raw_input → script.

    Reads: raw_input
    Writes: script
    """
    system_prompt = _load_prompt()

    async def data_processor_node(state: DocsTeamState) -> dict:
        raw_input = state.get("raw_input") or ""
        response = await model.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Process this input:\n\n{raw_input}"),
        ])
        return {"script": response.content}

    return data_processor_node


class DataProcessorAgent(BaseAgent):
    name = "data_processor"

    def build_graph(self) -> CompiledGraph:
        node = make_data_processor_node(self.adapter.get_model())
        graph = StateGraph(DocsTeamState)
        graph.add_node("process", node)
        graph.add_edge(START, "process")
        graph.add_edge("process", END)
        return graph.compile()
