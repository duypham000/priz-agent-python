from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage


class MemoryCondenser:
    def __init__(self, model: Any, max_messages: int = 20):
        self._model = model
        self._max_messages = max_messages

    def should_condense(self, messages: list[BaseMessage]) -> bool:
        return len(messages) > self._max_messages

    async def condense(self, messages: list[BaseMessage]) -> str:
        formatted = "\n".join(
            f"{msg.__class__.__name__}: {msg.content}" for msg in messages
        )
        prompt = f"Summarize this conversation history concisely:\n{formatted}"
        response = await self._model.ainvoke([HumanMessage(content=prompt)])
        return response.content
