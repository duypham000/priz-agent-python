from typing import Any, AsyncIterator

from langgraph.graph import END, START, StateGraph
from langgraph.pregel import Pregel as CompiledGraph
from langgraph.types import interrupt

from src.agents.base import BaseAgent
from src.agents.manager.planner import make_planner_node
from src.agents.manager.router import make_intent_classifier_node
from src.agents.manager.state import ManagerState
from src.agents.manager.validator import make_validator_node
from src.agents.teams.docs.supervisor import make_docs_team_node
from src.agents.teams.knowledge.supervisor import make_knowledge_team_node
from src.agents.teams.research.supervisor import make_research_team_node
from src.agents.teams.technical.supervisor import make_technical_team_node
from src.core.guardrails import GuardrailNode
from src.core.messages import get_last_human_message
from src.llm.base import BaseLLMAdapter


def _team_stub_node(state: ManagerState) -> dict:
    """Placeholder for teams not yet implemented (Phase 12-13)."""
    team = state.get("current_team") or "unknown"
    messages = state.get("messages", [])
    last_msg = get_last_human_message(messages)
    content = last_msg.content[:100] if last_msg else ""
    return {"team_output": f"[{team} team] Processed: {content}"}


def _hitl_node(state: ManagerState) -> dict:
    """HITL node: suspends graph execution until human approval is received."""
    feedback = interrupt({"reason": "Approval needed", "plan": state.get("plan")})
    return {
        "hitl_feedback": str(feedback) if feedback is not None else None,
        "hitl_required": False,
    }


def _route_after_guardrail(state: ManagerState) -> str:
    if state.get("final_response") is not None:
        return "end"
    return "continue"


def _route_after_team(state: ManagerState) -> str:
    if state.get("hitl_required"):
        return "hitl"
    return "validate"


def _make_team_execute_node(adapter: BaseLLMAdapter):
    """Return a team dispatch node that routes to the appropriate team implementation."""
    docs_node = make_docs_team_node(adapter)
    research_node = make_research_team_node(adapter)
    technical_node = make_technical_team_node(adapter)
    knowledge_node = make_knowledge_team_node(adapter)

    async def team_execute_node(state: ManagerState) -> dict:
        team = state.get("current_team") or "unknown"
        if team == "docs":
            return await docs_node(state)
        if team == "research":
            return await research_node(state)
        if team == "technical":
            return await technical_node(state)
        if team == "knowledge":
            return await knowledge_node(state)
        return _team_stub_node(state)

    return team_execute_node


class ManagerAgent(BaseAgent):
    name = "manager"

    def __init__(
        self,
        adapter: BaseLLMAdapter,
        checkpointer: Any = None,
        config: dict[str, Any] | None = None,
    ):
        super().__init__(adapter, config)
        self.checkpointer = checkpointer

    def build_graph(self) -> CompiledGraph:
        model = self.adapter.get_model()

        guardrail = GuardrailNode.default()
        intent_classifier = make_intent_classifier_node(model)
        planner = make_planner_node(model)
        validator = make_validator_node(model)
        team_execute = _make_team_execute_node(self.adapter)

        graph = StateGraph(ManagerState)
        graph.add_node("guardrail", guardrail)
        graph.add_node("intent_classifier", intent_classifier)
        graph.add_node("planner", planner)
        graph.add_node("team_execute", team_execute)
        graph.add_node("hitl", _hitl_node)
        graph.add_node("validator", validator)

        graph.add_edge(START, "guardrail")
        graph.add_conditional_edges(
            "guardrail",
            _route_after_guardrail,
            {"end": END, "continue": "intent_classifier"},
        )
        graph.add_edge("intent_classifier", "planner")
        graph.add_edge("planner", "team_execute")
        graph.add_conditional_edges(
            "team_execute",
            _route_after_team,
            {"hitl": "hitl", "validate": "validator"},
        )
        graph.add_edge("hitl", "validator")
        graph.add_edge("validator", END)

        return graph.compile(checkpointer=self.checkpointer)

    async def run(self, state: Any) -> Any:
        thread_id = state.get("thread_id", "default") if isinstance(state, dict) else "default"
        config = {**self.config, "configurable": {"thread_id": thread_id}}
        return await self._get_graph().ainvoke(state, config=config)

    async def stream(self, state: Any) -> AsyncIterator[dict[str, Any]]:
        thread_id = state.get("thread_id", "default") if isinstance(state, dict) else "default"
        config = {**self.config, "configurable": {"thread_id": thread_id}}
        async for chunk in self._get_graph().astream(state, config=config):
            yield chunk
