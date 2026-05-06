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
from src.agents.teams.research.state import ResearchTeamState

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM_PROMPT = (
    "You are a strategic advisor specializing in SWOT analysis and business intelligence. "
    "Given internal knowledge and market research, synthesize a comprehensive strategic recommendation. "
    "Structure your response as:\n"
    "## SWOT Analysis\n"
    "**Strengths:** ...\n"
    "**Weaknesses:** ...\n"
    "**Opportunities:** ...\n"
    "**Threats:** ...\n\n"
    "## Strategic Recommendation\n"
    "Provide actionable, prioritized recommendations based on the analysis."
)


def _load_prompt() -> str:
    path = Path(__file__).parents[4] / "prompts" / "teams" / "research" / "strategic_advisor.yaml"
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return data["system_prompt"]
    except Exception:
        logger.debug("strategic_advisor prompt not found, using default")
        return _DEFAULT_SYSTEM_PROMPT


def make_strategic_advisor_node(
    model: BaseChatModel,
) -> Callable[[ResearchTeamState], Coroutine[Any, Any, dict]]:
    """Return a LangGraph node that synthesizes internal + web data into SWOT + recommendation.

    Reads: query, internal_knowledge, web_results
    Writes: recommendation
    """
    system_prompt = _load_prompt()

    async def strategic_advisor_node(state: ResearchTeamState) -> dict:
        query = state.get("query") or ""
        internal = state.get("internal_knowledge") or "No internal knowledge available."
        web = state.get("web_results") or "No web research available."

        user_content = (
            f"Research Query: {query}\n\n"
            f"## Internal Knowledge Base\n{internal}\n\n"
            f"## Market Research (Web)\n{web}\n\n"
            "Based on both sources above, provide a SWOT analysis and strategic recommendation."
        )
        response = await model.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ])
        return {"recommendation": response.content}

    return strategic_advisor_node


class StrategicAdvisorAgent(BaseAgent):
    name = "strategic_advisor"

    def build_graph(self) -> CompiledGraph:
        node = make_strategic_advisor_node(self.adapter.get_model())
        graph = StateGraph(ResearchTeamState)
        graph.add_node("strategic_advisor", node)
        graph.add_edge(START, "strategic_advisor")
        graph.add_edge("strategic_advisor", END)
        return graph.compile()
