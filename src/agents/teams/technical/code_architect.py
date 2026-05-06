from __future__ import annotations

import logging
import re
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
    "You are a senior software engineer who generates clean, production-ready code "
    "from design specifications. Follow SOLID principles, use meaningful variable names, "
    "and include error handling. Output ONLY the code block with appropriate language tag."
)

_CODE_BLOCK_RE = re.compile(r"```(?:\w+)?\n(.*?)```", re.DOTALL)


def _load_prompt() -> str:
    path = Path(__file__).parents[4] / "prompts" / "teams" / "technical" / "code_architect.yaml"
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return data["system_prompt"]
    except Exception:
        logger.debug("code_architect prompt not found, using default")
        return _DEFAULT_SYSTEM_PROMPT


def _extract_code(response_text: str) -> str:
    """Extract raw code from a markdown code block, or return the full response if no block found."""
    match = _CODE_BLOCK_RE.search(response_text)
    if match:
        return match.group(1).strip()
    return response_text.strip()


def make_code_architect_node(
    model: BaseChatModel,
) -> Callable[[TechnicalTeamState], Coroutine[Any, Any, dict]]:
    """Return a LangGraph node that generates code from a design spec.

    Reads: design_spec
    Writes: code_output
    """
    system_prompt = _load_prompt()

    async def code_architect_node(state: TechnicalTeamState) -> dict:
        design_spec = state.get("design_spec") or "No design specification provided."

        response = await model.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=f"Generate implementation code for the following design specification:\n\n{design_spec}"
            ),
        ])
        code = _extract_code(response.content)
        return {"code_output": code}

    return code_architect_node


class CodeArchitectAgent(BaseAgent):
    name = "code_architect"

    def build_graph(self) -> CompiledGraph:
        node = make_code_architect_node(self.adapter.get_model())
        graph = StateGraph(TechnicalTeamState)
        graph.add_node("code_architect", node)
        graph.add_edge(START, "code_architect")
        graph.add_edge("code_architect", END)
        return graph.compile()
