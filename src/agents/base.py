from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from langgraph.pregel import Pregel as CompiledGraph

from src.core.state import AgentState
from src.llm.base import BaseLLMAdapter


class BaseAgent(ABC):
    name: str  # subclasses define as class-level attribute

    def __init__(
        self,
        adapter: BaseLLMAdapter,
        config: dict[str, Any] | None = None,
    ):
        self.adapter = adapter
        self.config = config or {}
        self._graph: CompiledGraph | None = None

    @abstractmethod
    def build_graph(self) -> CompiledGraph: ...

    def _get_graph(self) -> CompiledGraph:
        if self._graph is None:
            self._graph = self.build_graph()
        return self._graph

    async def run(self, state: AgentState) -> AgentState:
        return await self._get_graph().ainvoke(state, config=self.config)

    async def stream(self, state: AgentState) -> AsyncIterator[dict[str, Any]]:
        async for chunk in self._get_graph().astream(state, config=self.config):
            yield chunk
