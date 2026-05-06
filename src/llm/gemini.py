from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI

from src.llm.base import BaseLLMAdapter
from src.llm.token_counter import TokenCountProvider, TokenCounter
from src.settings import settings


class GeminiAdapter(BaseLLMAdapter):
    def __init__(
        self,
        model_id: str = "gemini-2.0-flash",
        temperature: float = 0.0,
        api_key: str | None = None,
    ):
        self._model_id = model_id
        self._temperature = temperature
        self._api_key = api_key or settings.gemini_api_key
        self._model: ChatGoogleGenerativeAI | None = None

    def get_model(self) -> BaseChatModel:
        if self._model is None:
            self._model = ChatGoogleGenerativeAI(
                model=self._model_id,
                temperature=self._temperature,
                google_api_key=self._api_key,
            )
        return self._model

    def count_tokens(self, text: str) -> int:
        if self._model is not None:
            try:
                return self._model.get_num_tokens(text)
            except Exception:
                pass
        return TokenCounter().count(text, TokenCountProvider.GEMINI)

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self._model_id
