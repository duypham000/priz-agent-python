from __future__ import annotations

from typing import Any, Callable, Coroutine

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.pregel import Pregel as CompiledGraph

from src.agents.base import BaseAgent
from src.agents.manager.state import ManagerState
from src.agents.teams.technical.code_architect import make_code_architect_node
from src.agents.teams.technical.quality_gatekeeper import make_quality_gatekeeper_node
from src.agents.teams.technical.state import TechnicalTeamState
from src.agents.teams.technical.visual_interpreter import make_visual_interpreter_node
from src.core.messages import get_last_human_message
from src.llm.base import BaseLLMAdapter


def _build_technical_pipeline(model: BaseChatModel) -> CompiledGraph:
    """Build the linear TechnicalTeam pipeline: visual_interpreter → code_architect → quality_gatekeeper."""
    graph = StateGraph(TechnicalTeamState)
    graph.add_node("visual_interpreter", make_visual_interpreter_node(model))
    graph.add_node("code_architect", make_code_architect_node(model))
    graph.add_node("quality_gatekeeper", make_quality_gatekeeper_node(model))
    graph.add_edge(START, "visual_interpreter")
    graph.add_edge("visual_interpreter", "code_architect")
    graph.add_edge("code_architect", "quality_gatekeeper")
    graph.add_edge("quality_gatekeeper", END)
    return graph.compile()


def _compose_output(result: dict) -> str:
    """Combine technical pipeline outputs into a single team_output string for ManagerState."""
    parts: list[str] = []
    if result.get("design_spec"):
        parts.append(f"## Design Specification\n{result['design_spec']}")
    if result.get("code_output"):
        parts.append(f"## Generated Code\n```\n{result['code_output']}\n```")
    if result.get("review_report"):
        parts.append(f"## Review Report\n{result['review_report']}")
    verdict = result.get("verdict")
    if verdict:
        parts.append(f"## Verdict\n{verdict}")
    return "\n\n".join(parts) if parts else "[TechnicalTeam] No output produced"


def make_technical_team_node(
    adapter: BaseLLMAdapter,
) -> Callable[[ManagerState], Coroutine[Any, Any, dict]]:
    """Return a ManagerState → dict node for use inside the manager graph.

    Bridges ManagerState → TechnicalTeamState, runs the full technical pipeline,
    then writes team_output back into ManagerState.
    """
    model = adapter.get_model()
    pipeline = _build_technical_pipeline(model)

    async def technical_team_node(state: ManagerState) -> dict:
        messages = state.get("messages", [])
        last_msg = get_last_human_message(messages)
        # Use the last human message as the initial design input
        # (could be a text description or a multimodal message with image)
        initial_messages = [last_msg] if last_msg else []

        technical_state: TechnicalTeamState = {
            "messages": initial_messages,
            "design_spec": None,
            "code_output": None,
            "review_report": None,
            "verdict": None,
        }

        result = await pipeline.ainvoke(technical_state)
        return {"team_output": _compose_output(result)}

    return technical_team_node


class TechnicalTeamSupervisor(BaseAgent):
    """Standalone technical team agent — wraps the pipeline for direct invocation."""

    name = "technical_supervisor"

    def build_graph(self) -> CompiledGraph:
        return _build_technical_pipeline(self.adapter.get_model())
