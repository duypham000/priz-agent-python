import asyncio
import hashlib
import random
from typing import Any, Iterator

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from src.core.exceptions import LLMProviderError
from src.llm.base import BaseLLMAdapter
from src.llm.token_counter import TokenCountProvider, TokenCounter


class _SeededFakeChatModel(BaseChatModel):
    seed: int = 42
    delay_ms: int = 0
    error_rate: float = 0.0
    responses: list[str] = Field(default_factory=lambda: ["mock response"])
    tool_calls_map: dict[str, list[dict]] = Field(default_factory=dict)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self._maybe_raise()
        response_text = self._pick_response(messages)
        tool_calls = self.tool_calls_map.get(response_text, [])
        msg = AIMessage(content=response_text, tool_calls=tool_calls)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.delay_ms > 0:
            await asyncio.sleep(self.delay_ms / 1000.0)
        self._maybe_raise()
        response_text = self._pick_response(messages)
        tool_calls = self.tool_calls_map.get(response_text, [])
        msg = AIMessage(content=response_text, tool_calls=tool_calls)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _pick_response(self, messages: list[BaseMessage]) -> str:
        last_content = messages[-1].content if messages else ""
        key = f"{self.seed}:{last_content}"
        idx = int(hashlib.md5(key.encode()).hexdigest(), 16) % len(self.responses)
        return self.responses[idx]

    def _maybe_raise(self) -> None:
        if self.error_rate > 0.0:
            rng = random.Random(self.seed)
            if rng.random() < self.error_rate:
                raise LLMProviderError(
                    "Simulated LLM error",
                    provider="mock",
                    code="MOCK_ERROR",
                )

    @property
    def _llm_type(self) -> str:
        return "seeded-fake-chat-model"


class MockAdapter(BaseLLMAdapter):
    def __init__(
        self,
        seed: int = 42,
        delay_ms: int = 0,
        error_rate: float = 0.0,
        responses: list[str] | None = None,
        tool_calls_map: dict[str, list[dict]] | None = None,
        model_id: str = "mock-model",
    ):
        self._seed = seed
        self._delay_ms = delay_ms
        self._error_rate = error_rate
        self._responses = responses or ["mock response"]
        self._tool_calls_map = tool_calls_map or {}
        self._model_id = model_id
        self._model: _SeededFakeChatModel | None = None

    def get_model(self) -> BaseChatModel:
        if self._model is None:
            self._model = _SeededFakeChatModel(
                seed=self._seed,
                delay_ms=self._delay_ms,
                error_rate=self._error_rate,
                responses=self._responses,
                tool_calls_map=self._tool_calls_map,
            )
        return self._model

    def count_tokens(self, text: str) -> int:
        return TokenCounter().count(text, TokenCountProvider.MOCK)

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return self._model_id
