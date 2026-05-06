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

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM_PROMPT = (
    "You are a task extraction specialist. "
    "Extract all concrete action items as a JSON array. "
    "Each element: {\"name\": str, \"deadline\": \"YYYY-MM-DD or TBD\", "
    "\"owner\": str, \"priority\": \"high|medium|low\"}. "
    "Output ONLY the JSON array."
)


def _load_prompt() -> str:
    path = Path(__file__).parents[4] / "prompts" / "teams" / "docs" / "task_architect.yaml"
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return data["system_prompt"]
    except Exception:
        logger.debug("task_architect prompt not found, using default")
        return _DEFAULT_SYSTEM_PROMPT


def _normalize_task(t: dict) -> dict:
    return {
        "name": str(t.get("name", "Unnamed task")),
        "deadline": str(t.get("deadline", "TBD")),
        "owner": str(t.get("owner", "Unassigned")),
        "priority": str(t.get("priority", "medium")),
    }


def _parse_tasks(text: str) -> list[dict]:
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [_normalize_task(t) for t in parsed if isinstance(t, dict)]
    except (json.JSONDecodeError, ValueError):
        pass
    return []


def make_task_architect_node(
    model: BaseChatModel,
) -> Callable[[DocsTeamState], Coroutine[Any, Any, dict]]:
    """Return a LangGraph node that extracts action items from summary/script.

    Reads: summary (fallback: script, raw_input)
    Writes: tasks
    """
    system_prompt = _load_prompt()

    async def task_architect_node(state: DocsTeamState) -> dict:
        content = state.get("summary") or state.get("script") or state.get("raw_input") or ""
        response = await model.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Extract tasks from:\n\n{content}"),
        ])
        return {"tasks": _parse_tasks(response.content)}

    return task_architect_node


class TaskArchitectAgent(BaseAgent):
    name = "task_architect"

    def build_graph(self) -> CompiledGraph:
        node = make_task_architect_node(self.adapter.get_model())
        graph = StateGraph(DocsTeamState)
        graph.add_node("extract", node)
        graph.add_edge(START, "extract")
        graph.add_edge("extract", END)
        return graph.compile()
