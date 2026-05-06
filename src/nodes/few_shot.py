from typing import Any, Callable, Coroutine

from langchain_core.messages import SystemMessage

from src.core.messages import get_last_human_message
from src.core.state import AgentState
from src.memory.long_term import LongTermMemory


def make_few_shot_node(
    memory: LongTermMemory,
    k: int = 3,
) -> Callable[[AgentState], Coroutine[Any, Any, dict]]:
    """Return a LangGraph node that retrieves few-shot examples from vector store and injects them as a SystemMessage.

    Reads:  messages (last HumanMessage used as retrieval query)
    Writes: messages (appends SystemMessage with examples), or {} if no docs found
    """

    async def few_shot_node(state: AgentState) -> dict:
        last_human = get_last_human_message(state["messages"])
        query = last_human.content if last_human else ""
        docs = await memory.retrieve(query, k=k)
        if not docs:
            return {}
        examples = "\n\n".join(
            f"Example {i + 1}:\n{doc.page_content}" for i, doc in enumerate(docs)
        )
        content = f"Few-shot examples retrieved from knowledge base:\n\n{examples}"
        return {"messages": [SystemMessage(content=content)]}

    return few_shot_node
