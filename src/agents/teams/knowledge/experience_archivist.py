from __future__ import annotations

import asyncio
import json
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
from src.settings import settings

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM_PROMPT = (
    "You are an AI experience analyst. Given agent execution traces and a technical summary, "
    "identify: (1) success patterns — what approaches worked well, (2) failure modes — what caused errors, "
    "(3) actionable lessons — concrete improvements for future runs. "
    "Format your output as Markdown with sections: ## Success Patterns, ## Failure Modes, ## Lessons Learned."
)

_MAX_SPANS = 20
_PHOENIX_TIMEOUT = 5


def _load_prompt() -> str:
    path = Path(__file__).parents[4] / "prompts" / "teams" / "knowledge" / "experience_archivist.yaml"
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return data["system_prompt"]
    except Exception:
        logger.debug("experience_archivist prompt not found, using default")
        return _DEFAULT_SYSTEM_PROMPT


def _fetch_phoenix_spans_sync(endpoint: str) -> list[dict]:
    """Fetch recent spans from Phoenix REST API (blocking — run in executor)."""
    url = f"{endpoint.rstrip('/')}/v1/spans"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=_PHOENIX_TIMEOUT) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    spans = body.get("data", []) if isinstance(body, dict) else body
    return spans[:_MAX_SPANS]


def _format_spans(spans: list[dict]) -> str:
    """Convert Phoenix span dicts to a readable string for LLM context."""
    if not spans:
        return "(no trace data available)"
    lines = []
    for i, span in enumerate(spans, 1):
        attrs = span.get("attributes", span)
        name = attrs.get("name") or span.get("name", f"span-{i}")
        status = attrs.get("status", {})
        status_code = status.get("code", "UNKNOWN") if isinstance(status, dict) else str(status)
        inp = str(attrs.get("input", {}).get("value", ""))[:200]
        out = str(attrs.get("output", {}).get("value", ""))[:200]
        lines.append(f"[{i}] {name} | status={status_code}\n  input: {inp}\n  output: {out}")
    return "\n".join(lines)


async def _fetch_phoenix_context() -> str:
    """Try to fetch Phoenix trace data; return formatted string or fallback message."""
    try:
        loop = asyncio.get_event_loop()
        spans = await loop.run_in_executor(
            None, _fetch_phoenix_spans_sync, settings.phoenix_endpoint
        )
        return _format_spans(spans)
    except Exception as exc:
        logger.debug("Phoenix unavailable (%s), using empty trace context", exc)
        return "(Phoenix traces unavailable — analysis based on documentation context only)"


def make_experience_archivist_node(
    model: BaseChatModel,
) -> Callable[[KnowledgeTeamState], Coroutine[Any, Any, dict]]:
    """Return a LangGraph node that extracts lessons from Phoenix traces.

    Reads: technical_summary, messages
    Writes: lessons
    """
    system_prompt = _load_prompt()

    async def experience_archivist_node(state: KnowledgeTeamState) -> dict:
        technical_summary = state.get("technical_summary") or "No technical summary available."
        trace_context = await _fetch_phoenix_context()

        response = await model.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=(
                    f"## Technical Summary\n{technical_summary}\n\n"
                    f"## Agent Execution Traces\n{trace_context}\n\n"
                    "Based on the above, extract lessons learned and success patterns."
                )
            ),
        ])
        return {"lessons": response.content}

    return experience_archivist_node


class ExperienceArchivistAgent(BaseAgent):
    name = "experience_archivist"

    def build_graph(self) -> CompiledGraph:
        node = make_experience_archivist_node(self.adapter.get_model())
        graph = StateGraph(KnowledgeTeamState)
        graph.add_node("experience_archivist", node)
        graph.add_edge(START, "experience_archivist")
        graph.add_edge("experience_archivist", END)
        return graph.compile()
