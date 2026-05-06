from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional

import yaml
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.pregel import Pregel as CompiledGraph

from src.agents.base import BaseAgent
from src.agents.teams.research.state import ResearchTeamState
from src.memory.long_term import LongTermMemory

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM_PROMPT = (
    "You are an internal knowledge specialist. "
    "Given a user query and retrieved documents from the knowledge base, "
    "synthesize the information into a clear, concise answer. "
    "Focus on accuracy and relevance. If the documents are insufficient, "
    "state clearly what is and isn't known."
)


def _load_prompt() -> str:
    path = Path(__file__).parents[4] / "prompts" / "teams" / "research" / "internal_brain.yaml"
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return data["system_prompt"]
    except Exception:
        logger.debug("internal_brain prompt not found, using default")
        return _DEFAULT_SYSTEM_PROMPT


def make_internal_brain_node(
    model: BaseChatModel,
    memory: Optional[LongTermMemory] = None,
) -> Callable[[ResearchTeamState], Coroutine[Any, Any, dict]]:
    """Return a LangGraph node that retrieves internal knowledge via Corrective RAG.

    Reads: query
    Writes: internal_knowledge
    """
    system_prompt = _load_prompt()

    async def internal_brain_node(state: ResearchTeamState) -> dict:
        query = state.get("query") or ""

        if not memory:
            return {"internal_knowledge": "No internal knowledge base configured."}

        try:
            docs = await memory.corrective_rag(query, model, k=5, relevance_threshold=0.5)
        except Exception as exc:
            logger.warning("corrective_rag failed: %s", exc)
            return {"internal_knowledge": "Internal knowledge base unavailable."}

        if not docs:
            return {"internal_knowledge": "No relevant internal knowledge found for this query."}

        context = "\n\n---\n\n".join(
            f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
            for doc in docs
        )
        response = await model.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=f"Query: {query}\n\nRetrieved knowledge:\n{context}\n\nSynthesize the above into a concise answer."
            ),
        ])
        return {"internal_knowledge": response.content}

    return internal_brain_node


class InternalBrainAgent(BaseAgent):
    name = "internal_brain"

    def __init__(self, adapter, memory: Optional[LongTermMemory] = None, config=None):
        super().__init__(adapter, config)
        self.memory = memory

    def build_graph(self) -> CompiledGraph:
        node = make_internal_brain_node(self.adapter.get_model(), self.memory)
        graph = StateGraph(ResearchTeamState)
        graph.add_node("internal_brain", node)
        graph.add_edge(START, "internal_brain")
        graph.add_edge("internal_brain", END)
        return graph.compile()
