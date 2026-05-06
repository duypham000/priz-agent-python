from typing import Any, Callable, Coroutine

from langchain_core.language_models.chat_models import BaseChatModel

from src.core.state import AgentState
from src.memory.condensation import MemoryCondenser


def make_summarization_node(
    model: BaseChatModel,
    max_messages: int = 20,
) -> Callable[[AgentState], Coroutine[Any, Any, dict]]:
    """Return a LangGraph node that condenses conversation messages into a concise summary.

    Writes: final_response
    """
    condenser = MemoryCondenser(model=model, max_messages=max_messages)

    async def summarization_node(state: AgentState) -> dict:
        summary = await condenser.condense(state["messages"])
        return {"final_response": summary}

    return summarization_node
