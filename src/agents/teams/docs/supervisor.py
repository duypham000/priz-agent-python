from __future__ import annotations

import json
from typing import Any, Callable, Coroutine

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.pregel import Pregel as CompiledGraph

from src.agents.base import BaseAgent
from src.agents.manager.state import ManagerState
from src.agents.teams.docs.data_processor import make_data_processor_node
from src.agents.teams.docs.state import DocsTeamState
from src.agents.teams.docs.summarizer import make_meeting_minutes_node
from src.agents.teams.docs.sync_manager import make_sync_manager_node
from src.agents.teams.docs.task_architect import make_task_architect_node
from src.core.messages import get_last_human_message
from src.llm.base import BaseLLMAdapter


def _build_docs_pipeline(model: BaseChatModel) -> CompiledGraph:
    """Build the linear DocsTeam pipeline: data_processor → summarizer → task_architect → sync_manager."""
    graph = StateGraph(DocsTeamState)
    graph.add_node("data_processor", make_data_processor_node(model))
    graph.add_node("summarizer", make_meeting_minutes_node(model))
    graph.add_node("task_architect", make_task_architect_node(model))
    graph.add_node("sync_manager", make_sync_manager_node(model))
    graph.add_edge(START, "data_processor")
    graph.add_edge("data_processor", "summarizer")
    graph.add_edge("summarizer", "task_architect")
    graph.add_edge("task_architect", "sync_manager")
    graph.add_edge("sync_manager", END)
    return graph.compile()


def _compose_output(result: dict) -> str:
    """Combine docs pipeline outputs into a single team_output string for ManagerState."""
    parts: list[str] = []
    if result.get("summary"):
        parts.append(result["summary"])
    if result.get("tasks"):
        parts.append(
            "\n## Action Items\n```json\n"
            + json.dumps(result["tasks"], indent=2, ensure_ascii=False)
            + "\n```"
        )
    if result.get("sync_status"):
        parts.append(f"\n{result['sync_status']}")
    return "\n".join(parts) if parts else "[DocsTeam] No output produced"


def make_docs_team_node(
    adapter: BaseLLMAdapter,
) -> Callable[[ManagerState], Coroutine[Any, Any, dict]]:
    """Return a ManagerState → dict node for use inside the manager graph.

    Bridges ManagerState → DocsTeamState, runs the full docs pipeline,
    then writes team_output back into ManagerState.
    """
    model = adapter.get_model()
    pipeline = _build_docs_pipeline(model)

    async def docs_team_node(state: ManagerState) -> dict:
        messages = state.get("messages", [])
        last_msg = get_last_human_message(messages)
        raw_input = last_msg.content if last_msg else ""

        docs_state: DocsTeamState = {
            "messages": messages,
            "raw_input": raw_input,
            "script": None,
            "summary": None,
            "tasks": None,
            "sync_status": None,
        }

        result = await pipeline.ainvoke(docs_state)
        return {"team_output": _compose_output(result)}

    return docs_team_node


class DocsTeamSupervisor(BaseAgent):
    """Standalone docs team agent — wraps the pipeline for direct invocation."""

    name = "docs_supervisor"

    def build_graph(self) -> CompiledGraph:
        return _build_docs_pipeline(self.adapter.get_model())
