import asyncio
import time

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

from src.core.exceptions import LLMProviderError
from src.core.state import AgentState
from src.llm.mock import MockAdapter
from src.llm.registry import LLMRegistry
from src.llm.token_counter import TokenCountProvider, TokenCounter, count_tokens

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_adapter() -> MockAdapter:
    return MockAdapter(seed=42, responses=["alpha", "beta", "gamma"])


@pytest.fixture
def registry_with_mock(mock_adapter: MockAdapter) -> LLMRegistry:
    reg = LLMRegistry()
    reg.register_as("mock", mock_adapter)
    return reg


# ---------------------------------------------------------------------------
# TestMockAdapterDeterminism
# ---------------------------------------------------------------------------
class TestMockAdapterDeterminism:
    def test_same_input_same_seed_returns_same_response(self):
        # Arrange
        adapter = MockAdapter(seed=42, responses=["x", "y", "z"])
        model = adapter.get_model()
        messages = [HumanMessage(content="hello")]

        # Act
        result1 = model.invoke(messages)
        result2 = model.invoke(messages)

        # Assert
        assert result1.content == result2.content

    def test_different_seed_may_return_different_response(self):
        # Arrange
        responses = ["a", "b", "c", "d", "e", "f", "g", "h"]
        adapter1 = MockAdapter(seed=1, responses=responses)
        adapter2 = MockAdapter(seed=99, responses=responses)
        messages = [HumanMessage(content="test")]

        # Act
        r1 = adapter1.get_model().invoke(messages).content
        r2 = adapter2.get_model().invoke(messages).content

        # Assert — with 8 responses, seeds 1 and 99 should hash to different indexes
        assert r1 != r2

    def test_response_cycles_deterministically_by_input_hash(self):
        # Arrange
        adapter = MockAdapter(seed=42, responses=["alpha", "beta", "gamma"])
        model = adapter.get_model()

        # Act — same message always maps to same response
        r_hello_1 = model.invoke([HumanMessage(content="hello")]).content
        r_hello_2 = model.invoke([HumanMessage(content="hello")]).content
        r_world = model.invoke([HumanMessage(content="world")]).content

        # Assert
        assert r_hello_1 == r_hello_2
        assert r_hello_1 in {"alpha", "beta", "gamma"}
        assert r_world in {"alpha", "beta", "gamma"}


# ---------------------------------------------------------------------------
# TestMockAdapterLatency
# ---------------------------------------------------------------------------
class TestMockAdapterLatency:
    async def test_zero_delay_returns_quickly(self):
        # Arrange
        adapter = MockAdapter(delay_ms=0)
        model = adapter.get_model()

        # Act
        start = time.monotonic()
        await model.ainvoke([HumanMessage(content="hi")])
        elapsed_ms = (time.monotonic() - start) * 1000

        # Assert
        assert elapsed_ms < 500

    async def test_configured_delay_is_applied(self):
        # Arrange
        adapter = MockAdapter(delay_ms=150)
        model = adapter.get_model()

        # Act
        start = time.monotonic()
        await model.ainvoke([HumanMessage(content="hi")])
        elapsed_ms = (time.monotonic() - start) * 1000

        # Assert
        assert elapsed_ms >= 100

    def test_delay_does_not_affect_sync_generate(self):
        # Arrange — sync path (_generate) ignores delay_ms
        adapter = MockAdapter(delay_ms=5000)
        model = adapter.get_model()

        # Act
        start = time.monotonic()
        model.invoke([HumanMessage(content="hi")])
        elapsed_ms = (time.monotonic() - start) * 1000

        # Assert — sync call should complete well under 1s even with delay_ms=5000
        assert elapsed_ms < 1000


# ---------------------------------------------------------------------------
# TestMockAdapterErrorSimulation
# ---------------------------------------------------------------------------
class TestMockAdapterErrorSimulation:
    def test_error_rate_zero_never_raises(self):
        # Arrange
        adapter = MockAdapter(error_rate=0.0, seed=42)
        model = adapter.get_model()

        # Act / Assert — 10 calls, none should raise
        for _ in range(10):
            result = model.invoke([HumanMessage(content="ok")])
            assert result is not None

    def test_error_rate_one_always_raises(self):
        # Arrange
        adapter = MockAdapter(error_rate=1.0, seed=42)
        model = adapter.get_model()

        # Act / Assert
        with pytest.raises(LLMProviderError):
            model.invoke([HumanMessage(content="fail")])

    def test_error_is_llm_provider_error(self):
        # Arrange
        adapter = MockAdapter(error_rate=1.0, seed=42)
        model = adapter.get_model()

        # Act
        try:
            model.invoke([HumanMessage(content="fail")])
        except Exception as exc:
            # Assert
            assert isinstance(exc, LLMProviderError)

    def test_error_rate_partial_is_deterministic_with_seed(self):
        # Arrange — same seed → same raise/no-raise pattern
        adapter1 = MockAdapter(error_rate=0.5, seed=7)
        adapter2 = MockAdapter(error_rate=0.5, seed=7)

        # Act
        result1_raises = False
        result2_raises = False
        try:
            adapter1.get_model().invoke([HumanMessage(content="x")])
        except LLMProviderError:
            result1_raises = True
        try:
            adapter2.get_model().invoke([HumanMessage(content="x")])
        except LLMProviderError:
            result2_raises = True

        # Assert — same seed → same outcome
        assert result1_raises == result2_raises


# ---------------------------------------------------------------------------
# TestMockAdapterToolCalls
# ---------------------------------------------------------------------------
class TestMockAdapterToolCalls:
    def test_no_tool_calls_by_default(self):
        # Arrange
        adapter = MockAdapter(responses=["mock response"])
        model = adapter.get_model()

        # Act
        result = model.invoke([HumanMessage(content="hi")])

        # Assert
        assert isinstance(result, AIMessage)
        assert result.tool_calls == []

    def test_tool_calls_returned_when_mapped(self):
        # Arrange
        tool_def = [{"name": "search", "args": {"query": "test"}, "id": "call-1"}]
        adapter = MockAdapter(
            responses=["mock response"],
            tool_calls_map={"mock response": tool_def},
        )
        model = adapter.get_model()

        # Act
        result = model.invoke([HumanMessage(content="x")])

        # Assert — "mock response" is the only response, so it always maps
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "search"

    def test_tool_call_structure_is_valid_for_langgraph(self):
        # Arrange
        tool_def = [{"name": "lookup", "args": {"id": 1}, "id": "tc-abc"}]
        adapter = MockAdapter(
            responses=["mock response"],
            tool_calls_map={"mock response": tool_def},
        )

        # Act
        result = adapter.get_model().invoke([HumanMessage(content="go")])

        # Assert — each tool_call must have name, args, id (LangGraph requirement)
        for tc in result.tool_calls:
            assert "name" in tc
            assert "args" in tc
            assert "id" in tc

    def test_multiple_tool_calls_returned(self):
        # Arrange
        tool_defs = [
            {"name": "tool_a", "args": {}, "id": "id-1"},
            {"name": "tool_b", "args": {"x": 2}, "id": "id-2"},
        ]
        adapter = MockAdapter(
            responses=["mock response"],
            tool_calls_map={"mock response": tool_defs},
        )

        # Act
        result = adapter.get_model().invoke([HumanMessage(content="multi")])

        # Assert
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0]["name"] == "tool_a"
        assert result.tool_calls[1]["name"] == "tool_b"


# ---------------------------------------------------------------------------
# TestMockAdapterLangGraphCompatibility
# ---------------------------------------------------------------------------
class TestMockAdapterLangGraphCompatibility:
    def test_get_model_returns_base_chat_model(self, mock_adapter: MockAdapter):
        # Arrange / Act
        model = mock_adapter.get_model()

        # Assert
        assert isinstance(model, BaseChatModel)

    async def test_model_can_be_used_in_langgraph_state_graph(self):
        # Arrange
        from langgraph.graph import StateGraph

        adapter = MockAdapter(seed=42, responses=["hello from mock"])

        async def chat_node(state: AgentState) -> dict:
            response = await adapter.get_model().ainvoke(state["messages"])
            return {"messages": [response]}

        graph = StateGraph(AgentState)
        graph.add_node("chat", chat_node)
        graph.set_entry_point("chat")
        graph.set_finish_point("chat")
        compiled = graph.compile()

        state: AgentState = {
            "thread_id": "t1",
            "user_id": "u1",
            "messages": [HumanMessage(content="hi")],
            "intent": None,
            "plan": None,
            "current_team": None,
            "team_output": None,
            "hitl_required": False,
            "final_response": None,
        }

        # Act
        result = await compiled.ainvoke(state)

        # Assert
        assert "messages" in result
        assert len(result["messages"]) > 0

    async def test_compiled_graph_produces_ai_message(self):
        # Arrange
        from langgraph.graph import StateGraph

        adapter = MockAdapter(seed=42, responses=["ai says hi"])

        async def chat_node(state: AgentState) -> dict:
            response = await adapter.get_model().ainvoke(state["messages"])
            return {"messages": [response]}

        graph = StateGraph(AgentState)
        graph.add_node("chat", chat_node)
        graph.set_entry_point("chat")
        graph.set_finish_point("chat")
        compiled = graph.compile()

        state: AgentState = {
            "thread_id": "t2",
            "user_id": "u1",
            "messages": [HumanMessage(content="test")],
            "intent": None,
            "plan": None,
            "current_team": None,
            "team_output": None,
            "hitl_required": False,
            "final_response": None,
        }

        # Act
        result = await compiled.ainvoke(state)
        last_message = result["messages"][-1]

        # Assert
        assert isinstance(last_message, AIMessage)

    def test_get_model_cached_returns_same_instance(self, mock_adapter: MockAdapter):
        # Arrange / Act
        model1 = mock_adapter.get_model()
        model2 = mock_adapter.get_model()

        # Assert — same object, lazy singleton
        assert model1 is model2


# ---------------------------------------------------------------------------
# TestMockAdapterCountTokens
# ---------------------------------------------------------------------------
class TestMockAdapterCountTokens:
    def test_count_tokens_returns_integer(self, mock_adapter: MockAdapter):
        # Arrange / Act
        result = mock_adapter.count_tokens("hello world")

        # Assert
        assert isinstance(result, int)

    def test_count_tokens_empty_string(self, mock_adapter: MockAdapter):
        # Arrange / Act
        result = mock_adapter.count_tokens("")

        # Assert
        assert result == 0

    def test_count_tokens_longer_text_returns_more(self, mock_adapter: MockAdapter):
        # Arrange / Act
        short = mock_adapter.count_tokens("one")
        long = mock_adapter.count_tokens("one two three four five")

        # Assert
        assert long > short


# ---------------------------------------------------------------------------
# TestLLMRegistry
# ---------------------------------------------------------------------------
class TestLLMRegistry:
    def test_register_and_get_by_model_name(self):
        # Arrange
        adapter = MockAdapter(model_id="my-mock")
        reg = LLMRegistry()

        # Act
        reg.register(adapter)
        result = reg.get("my-mock")

        # Assert
        assert result is adapter

    def test_register_as_alias_and_get(self):
        # Arrange
        adapter = MockAdapter()
        reg = LLMRegistry()

        # Act
        reg.register_as("default", adapter)
        result = reg.get("default")

        # Assert
        assert result is adapter

    def test_get_unknown_raises_llm_provider_error(self):
        # Arrange
        reg = LLMRegistry()

        # Act / Assert
        with pytest.raises(LLMProviderError) as exc_info:
            reg.get("unknown")
        assert "ADAPTER_NOT_FOUND" in str(exc_info.value.code)

    def test_list_models_returns_all_registered(self):
        # Arrange
        reg = LLMRegistry()
        a1 = MockAdapter(model_id="m1")
        a2 = MockAdapter(model_id="m2")

        # Act
        reg.register(a1)
        reg.register(a2)
        models = reg.list_models()

        # Assert
        assert "m1" in models
        assert "m2" in models

    def test_get_model_convenience_method(self, registry_with_mock: LLMRegistry):
        # Arrange / Act
        model = registry_with_mock.get_model("mock")

        # Assert
        assert isinstance(model, BaseChatModel)

    def test_fallback_chain_skips_failed_primary(self):
        # Arrange
        class _FailingAdapter(MockAdapter):
            def get_model(self):
                raise RuntimeError("intentional failure")

        reg = LLMRegistry()
        failing = _FailingAdapter(model_id="failing")
        fallback = MockAdapter(model_id="fallback-mock")
        reg.register(failing)
        reg.register(fallback)

        # Act
        result = reg.get_with_fallback("failing")

        # Assert
        assert result is fallback

    def test_set_fallback_chain_reorders_priority(self):
        # Arrange
        reg = LLMRegistry()
        a = MockAdapter(model_id="a")
        b = MockAdapter(model_id="b")
        reg.register(a)
        reg.register(b)

        # Act
        reg.set_fallback_chain(["b", "a"])

        # Assert — chain order is honoured
        assert reg._fallback_chain[0] == "b"
        assert reg._fallback_chain[1] == "a"

    def test_count_tokens_delegates_to_adapter(self, registry_with_mock: LLMRegistry, mock_adapter: MockAdapter):
        # Arrange
        text = "hello world test"

        # Act
        registry_result = registry_with_mock.count_tokens("mock", text)
        adapter_result = mock_adapter.count_tokens(text)

        # Assert
        assert registry_result == adapter_result


# ---------------------------------------------------------------------------
# TestTokenCounter
# ---------------------------------------------------------------------------
class TestTokenCounter:
    def test_generic_count_is_word_based(self):
        # Arrange
        counter = TokenCounter()

        # Act
        result = counter.count("hello world", TokenCountProvider.GENERIC)

        # Assert
        assert result == 2

    def test_gemini_heuristic_nonzero_for_nonempty(self):
        # Arrange
        counter = TokenCounter()

        # Act
        result = counter.count("hello world", TokenCountProvider.GEMINI)

        # Assert
        assert result > 0

    def test_mock_count_is_word_count(self):
        # Arrange
        counter = TokenCounter()

        # Act
        result = counter.count("one two three four", TokenCountProvider.MOCK)

        # Assert
        assert result == 4

    def test_empty_string_returns_zero(self):
        # Arrange
        counter = TokenCounter()

        # Act / Assert
        assert counter.count("", TokenCountProvider.GENERIC) == 0
        assert counter.count("", TokenCountProvider.GEMINI) == 0
        assert counter.count("", TokenCountProvider.MOCK) == 0

    def test_convenience_function_string_provider(self):
        # Arrange / Act
        result = count_tokens("hello world", "gemini")

        # Assert
        assert result > 0
