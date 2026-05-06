from __future__ import annotations

from typing import Any, Callable, Coroutine, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.pregel import Pregel as CompiledGraph

from src.agents.base import BaseAgent
from src.agents.manager.state import ManagerState
from src.agents.teams.research.internal_brain import make_internal_brain_node
from src.agents.teams.research.market_scout import make_market_scout_node
from src.agents.teams.research.state import ResearchTeamState
from src.agents.teams.research.strategic_advisor import make_strategic_advisor_node
from src.core.messages import get_last_human_message
from src.llm.base import BaseLLMAdapter
from src.memory.long_term import LongTermMemory


def _build_research_pipeline(
    model: BaseChatModel,
    memory: Optional[LongTermMemory] = None,
) -> CompiledGraph:
    """Build the linear ResearchTeam pipeline: internal_brain → market_scout → strategic_advisor."""
    graph = StateGraph(ResearchTeamState)
    graph.add_node("internal_brain", make_internal_brain_node(model, memory))
    graph.add_node("market_scout", make_market_scout_node(model))
    graph.add_node("strategic_advisor", make_strategic_advisor_node(model))
    graph.add_edge(START, "internal_brain")
    graph.add_edge("internal_brain", "market_scout")
    graph.add_edge("market_scout", "strategic_advisor")
    graph.add_edge("strategic_advisor", END)
    return graph.compile()


def _compose_output(result: dict) -> str:
    """Combine research pipeline outputs into a single team_output string for ManagerState."""
    parts: list[str] = []
    if result.get("internal_knowledge"):
        parts.append(f"## Internal Knowledge\n{result['internal_knowledge']}")
    if result.get("web_results"):
        parts.append(f"## Market Research\n{result['web_results']}")
    if result.get("recommendation"):
        parts.append(f"## Strategic Advisory\n{result['recommendation']}")
    return "\n\n".join(parts) if parts else "[ResearchTeam] No output produced"


def make_research_team_node(
    adapter: BaseLLMAdapter,
    memory: Optional[LongTermMemory] = None,
) -> Callable[[ManagerState], Coroutine[Any, Any, dict]]:
    """Return a ManagerState → dict node for use inside the manager graph.

    Bridges ManagerState → ResearchTeamState, runs the full research pipeline,
    then writes team_output back into ManagerState.
    """
    model = adapter.get_model()
    pipeline = _build_research_pipeline(model, memory)

    async def research_team_node(state: ManagerState) -> dict:
        messages = state.get("messages", [])
        last_msg = get_last_human_message(messages)
        query = last_msg.content if last_msg else ""

        research_state: ResearchTeamState = {
            "messages": messages,
            "query": query,
            "internal_knowledge": None,
            "web_results": None,
            "recommendation": None,
        }

        result = await pipeline.ainvoke(research_state)
        return {"team_output": _compose_output(result)}

    return research_team_node


class ResearchTeamSupervisor(BaseAgent):
    """Standalone research team agent — wraps the pipeline for direct invocation."""

    name = "research_supervisor"

    def __init__(
        self,
        adapter: BaseLLMAdapter,
        memory: Optional[LongTermMemory] = None,
        config=None,
    ):
        super().__init__(adapter, config)
        self.memory = memory

    def build_graph(self) -> CompiledGraph:
        return _build_research_pipeline(self.adapter.get_model(), self.memory)
