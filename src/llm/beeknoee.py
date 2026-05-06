import json
from typing import Any

import httpx
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from src.core.exceptions import LLMProviderError
from src.llm.base import BaseLLMAdapter
from src.llm.token_counter import TokenCountProvider, TokenCounter
from src.settings import settings

_ROLE_MAP = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}


class _BeeknoeeHttpChatModel(BaseChatModel):
    base_url: str
    api_key: str
    model_id: str
    temperature: float = 0.0
    timeout: int = 30

    def _messages_to_openai(self, messages: list[BaseMessage]) -> list[dict]:
        return [
            {"role": _ROLE_MAP.get(m.type, "user"), "content": m.content}
            for m in messages
        ]

    def _parse_response(self, data: dict) -> AIMessage:
        choice = data["choices"][0]["message"]
        content = choice.get("content") or ""
        tool_calls_raw = choice.get("tool_calls") or []
        tool_calls = [
            {
                "name": tc["function"]["name"],
                "args": json.loads(tc["function"]["arguments"]),
                "id": tc["id"],
            }
            for tc in tool_calls_raw
        ]
        return AIMessage(content=content, tool_calls=tool_calls)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload = {
            "model": self.model_id,
            "messages": self._messages_to_openai(messages),
            "temperature": self.temperature,
        }
        try:
            resp = httpx.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            msg = self._parse_response(resp.json())
        except httpx.HTTPError as e:
            raise LLMProviderError(str(e), provider="beeknoee") from e
        return ChatResult(generations=[ChatGeneration(message=msg)])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload = {
            "model": self.model_id,
            "messages": self._messages_to_openai(messages),
            "temperature": self.temperature,
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                resp.raise_for_status()
                msg = self._parse_response(resp.json())
        except httpx.HTTPError as e:
            raise LLMProviderError(str(e), provider="beeknoee") from e
        return ChatResult(generations=[ChatGeneration(message=msg)])

    @property
    def _llm_type(self) -> str:
        return "beeknoee-http-chat-model"


class BeeknoeeAdapter(BaseLLMAdapter):
    def __init__(
        self,
        model_id: str = "beeknoee-default",
        temperature: float = 0.0,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        self._model_id = model_id
        self._temperature = temperature
        self._base_url = base_url or settings.beeknoee_base_url
        self._api_key = api_key or settings.beeknoee_api_key
        self._model: _BeeknoeeHttpChatModel | None = None

    def get_model(self) -> BaseChatModel:
        if self._model is None:
            self._model = _BeeknoeeHttpChatModel(
                base_url=self._base_url,
                api_key=self._api_key,
                model_id=self._model_id,
                temperature=self._temperature,
            )
        return self._model

    def count_tokens(self, text: str) -> int:
        return TokenCounter().count(text, TokenCountProvider.BEEKNOEE)

    @property
    def provider_name(self) -> str:
        return "beeknoee"

    @property
    def model_name(self) -> str:
        return self._model_id
