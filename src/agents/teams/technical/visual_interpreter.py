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
from src.agents.teams.technical.state import TechnicalTeamState

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM_PROMPT = (
    "You are a UI/UX design interpreter specializing in translating visual designs "
    "and Figma specifications into precise technical documentation. "
    "Analyze the provided design and output a structured spec including: "
    "Component hierarchy, Layout (grid/flex), Colors, Typography, Interactions, and Data requirements."
)


def _load_prompt() -> str:
    path = Path(__file__).parents[4] / "prompts" / "teams" / "technical" / "visual_interpreter.yaml"
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return data["system_prompt"]
    except Exception:
        logger.debug("visual_interpreter prompt not found, using default")
        return _DEFAULT_SYSTEM_PROMPT


def _extract_input(state: TechnicalTeamState) -> str:
    """Extract the design input from messages — text description or multimodal content."""
    messages = state.get("messages") or []
    for msg in reversed(messages):
        content = msg.content
        if isinstance(content, list):
            # Multimodal: list of dicts with type "text" or "image_url"
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(item.get("text", ""))
                    elif item.get("type") == "image_url":
                        url = item.get("image_url", {})
                        if isinstance(url, dict):
                            parts.append(f"[Image: {url.get('url', 'provided')}]")
                        else:
                            parts.append(f"[Image: {url}]")
            if parts:
                return "\n".join(parts)
        elif isinstance(content, str) and content.strip():
            return content.strip()
    return ""


def make_visual_interpreter_node(
    model: BaseChatModel,
) -> Callable[[TechnicalTeamState], Coroutine[Any, Any, dict]]:
    """Return a LangGraph node that interprets visual/Figma input into a design spec.

    Reads: messages (text or multimodal image content)
    Writes: design_spec
    """
    system_prompt = _load_prompt()

    async def visual_interpreter_node(state: TechnicalTeamState) -> dict:
        raw_input = _extract_input(state)

        if not raw_input:
            return {"design_spec": "No design input provided."}

        response = await model.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=f"Please analyze this design and produce a structured technical specification:\n\n{raw_input}"
            ),
        ])
        return {"design_spec": response.content}

    return visual_interpreter_node


class VisualInterpreterAgent(BaseAgent):
    name = "visual_interpreter"

    def build_graph(self) -> CompiledGraph:
        node = make_visual_interpreter_node(self.adapter.get_model())
        graph = StateGraph(TechnicalTeamState)
        graph.add_node("visual_interpreter", node)
        graph.add_edge(START, "visual_interpreter")
        graph.add_edge("visual_interpreter", END)
        return graph.compile()
