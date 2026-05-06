from __future__ import annotations

from typing import Any, Callable, Coroutine

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.pregel import Pregel as CompiledGraph

from src.agents.base import BaseAgent
from src.agents.manager.state import ManagerState
from src.agents.teams.knowledge.experience_archivist import make_experience_archivist_node
from src.agents.teams.knowledge.knowledge_scout import make_knowledge_scout_node
from src.agents.teams.knowledge.state import KnowledgeTeamState
from src.agents.teams.knowledge.yaml_architect import make_yaml_architect_node
from src.core.messages import get_last_human_message
from src.llm.base import BaseLLMAdapter


def _build_knowledge_pipeline(model: BaseChatModel) -> CompiledGraph:
    """Build the linear KnowledgeTeam pipeline: knowledge_scout → experience_archivist → yaml_architect."""
    graph = StateGraph(KnowledgeTeamState)
    graph.add_node("knowledge_scout", make_knowledge_scout_node(model))
    graph.add_node("experience_archivist", make_experience_archivist_node(model))
    graph.add_node("yaml_architect", make_yaml_architect_node(model))
    graph.add_edge(START, "knowledge_scout")
    graph.add_edge("knowledge_scout", "experience_archivist")
    graph.add_edge("experience_archivist", "yaml_architect")
    graph.add_edge("yaml_architect", END)
    return graph.compile()


def _compose_output(result: dict) -> str:
    """Combine knowledge pipeline outputs into a single team_output string for ManagerState."""
    parts: list[str] = []
    if result.get("technical_summary"):
        parts.append(f"## Technical Summary\n{result['technical_summary']}")
    if result.get("lessons"):
        parts.append(f"## Lessons Learned\n{result['lessons']}")
    if result.get("yaml_output"):
        parts.append(f"## Generated YAML Guidelines\n```yaml\n{result['yaml_output']}\n```")
    return "\n\n".join(parts) if parts else "[KnowledgeTeam] No output produced"


def make_knowledge_team_node(
    adapter: BaseLLMAdapter,
) -> Callable[[ManagerState], Coroutine[Any, Any, dict]]:
    """Return a ManagerState → dict node for use inside the manager graph.

    Bridges ManagerState → KnowledgeTeamState, runs the full knowledge pipeline,
    then writes team_output back into ManagerState.
    """
    model = adapter.get_model()
    pipeline = _build_knowledge_pipeline(model)

    async def knowledge_team_node(state: ManagerState) -> dict:
        messages = state.get("messages", [])
        last_msg = get_last_human_message(messages)
        initial_messages = [last_msg] if last_msg else []

        knowledge_state: KnowledgeTeamState = {
            "messages": initial_messages,
            "source_docs": None,
            "technical_summary": None,
            "lessons": None,
            "yaml_output": None,
        }

        result = await pipeline.ainvoke(knowledge_state)
        return {"team_output": _compose_output(result)}

    return knowledge_team_node


class KnowledgeTeamSupervisor(BaseAgent):
    """Standalone knowledge team agent — wraps the pipeline for direct invocation."""

    name = "knowledge_supervisor"

    def build_graph(self) -> CompiledGraph:
        return _build_knowledge_pipeline(self.adapter.get_model())
