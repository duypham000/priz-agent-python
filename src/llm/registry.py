from langchain_core.language_models.chat_models import BaseChatModel

from src.core.exceptions import LLMProviderError
from src.llm.base import BaseLLMAdapter


class LLMRegistry:
    """Central registry for LLM adapters. Agents must not import provider SDKs directly."""

    def __init__(self) -> None:
        self._adapters: dict[str, BaseLLMAdapter] = {}
        self._fallback_chain: list[str] = []

    def register(self, adapter: BaseLLMAdapter, *, primary: bool = False) -> None:
        """Register adapter using adapter.model_name as key."""
        key = adapter.model_name
        self._adapters[key] = adapter
        if primary:
            self._fallback_chain.insert(0, key)
        elif key not in self._fallback_chain:
            self._fallback_chain.append(key)

    def register_as(self, name: str, adapter: BaseLLMAdapter) -> None:
        """Register adapter under an alias (e.g. 'default', 'research')."""
        self._adapters[name] = adapter
        if name not in self._fallback_chain:
            self._fallback_chain.append(name)

    def get(self, name: str) -> BaseLLMAdapter:
        if name not in self._adapters:
            raise LLMProviderError(
                f"No adapter registered under name '{name}'",
                provider=name,
                code="ADAPTER_NOT_FOUND",
            )
        return self._adapters[name]

    def get_model(self, name: str) -> BaseChatModel:
        return self.get(name).get_model()

    def get_with_fallback(self, name: str) -> BaseLLMAdapter:
        """Try primary adapter; on failure walk the fallback chain."""
        try:
            adapter = self.get(name)
            adapter.get_model()
            return adapter
        except Exception:
            for fallback_name in self._fallback_chain:
                if fallback_name == name:
                    continue
                try:
                    adapter = self._adapters[fallback_name]
                    adapter.get_model()
                    return adapter
                except Exception:
                    continue
        raise LLMProviderError(
            f"All adapters in fallback chain failed for '{name}'",
            provider=name,
            code="ALL_ADAPTERS_FAILED",
        )

    def list_models(self) -> list[str]:
        return list(self._adapters.keys())

    def set_fallback_chain(self, chain: list[str]) -> None:
        self._fallback_chain = chain

    def count_tokens(self, name: str, text: str) -> int:
        return self.get(name).count_tokens(text)
