from abc import ABC, abstractmethod
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage

from src.core.caching import LLMCache


class BaseLLMAdapter(ABC):
    @abstractmethod
    def get_model(self) -> BaseChatModel:
        """Return the cached BaseChatModel instance. Agents call ainvoke() on the result."""
        ...

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens for this provider. No DB writes, no API calls required."""
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...

    async def ainvoke(
        self,
        messages: list[BaseMessage],
        cache: Optional[LLMCache] = None,
    ) -> AIMessage:
        """Invoke the model with optional semantic caching."""
        if cache is not None:
            key = cache.make_key(str(messages), self.model_name)
            cached = await cache.get(key)
            if cached is not None:
                return AIMessage(content=cached)

        result = await self.get_model().ainvoke(messages)

        if cache is not None and isinstance(result.content, str):
            await cache.set(key, result.content)

        return result
