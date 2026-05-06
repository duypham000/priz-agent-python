from __future__ import annotations

import json
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
from src.core.exceptions import ToolError
from src.tools.builtin.calendar import create_task
from src.settings import settings

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM_PROMPT = (
    "You are a task synchronization coordinator. "
    "Given task sync results, produce a concise Sync Status Report. "
    "Format: ## Sync Status Report, Total tasks: N, Synced: N, Failed: N. "
    "Output ONLY the status report."
)


def _load_prompt() -> str:
    path = Path(__file__).parents[4] / "prompts" / "teams" / "docs" / "sync_manager.yaml"
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return data["system_prompt"]
    except Exception:
        logger.debug("sync_manager prompt not found, using default")
        return _DEFAULT_SYSTEM_PROMPT


def _format_results_for_llm(results: list[dict], backend: str) -> str:
    total = len(results)
    synced = sum(1 for r in results if r["status"] == "synced")
    failed = total - synced
    failed_names = [r["task"] for r in results if r["status"] == "failed"]
    lines = [
        f"Backend: {backend}",
        f"Total tasks: {total}",
        f"Synced: {synced}",
        f"Failed: {failed}",
    ]
    if failed_names:
        lines.append("Failed tasks: " + ", ".join(failed_names))
    return "\n".join(lines)


def make_sync_manager_node(
    model: BaseChatModel,
) -> Callable[[DocsTeamState], Coroutine[Any, Any, dict]]:
    """Return a LangGraph node that syncs tasks to calendar backend.

    Reads: tasks
    Writes: sync_status
    """
    system_prompt = _load_prompt()

    async def sync_manager_node(state: DocsTeamState) -> dict:
        tasks = state.get("tasks") or []
        results: list[dict] = []

        for task in tasks:
            try:
                result = await create_task(
                    title=task.get("name", ""),
                    deadline=task.get("deadline", "TBD"),
                    owner=task.get("owner", "Unassigned"),
                )
                results.append({"task": task.get("name", ""), "status": "synced", "result": result})
            except ToolError as exc:
                logger.warning("Failed to sync task %r: %s", task.get("name"), exc)
                results.append({"task": task.get("name", ""), "status": "failed", "error": str(exc)})

        summary_text = _format_results_for_llm(results, settings.calendar_backend)
        response = await model.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=summary_text),
        ])
        return {"sync_status": response.content}

    return sync_manager_node


class SyncManagerAgent(BaseAgent):
    name = "sync_manager"

    def build_graph(self) -> CompiledGraph:
        node = make_sync_manager_node(self.adapter.get_model())
        graph = StateGraph(DocsTeamState)
        graph.add_node("sync", node)
        graph.add_edge(START, "sync")
        graph.add_edge("sync", END)
        return graph.compile()
