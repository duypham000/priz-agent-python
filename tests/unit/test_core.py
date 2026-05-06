import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.core.caching import LLMCache, RedisLLMCache
from src.core.conflict import ConflictResolver, ConflictStrategy
from src.core.exceptions import (
    AgentError,
    HitlRequiredException,
    LLMProviderError,
    QuotaExceededError,
    ToolError,
)
from src.core.guardrails import GuardrailNode
from src.core.messages import ai, filter_by_type, get_last_human_message, human, system, tool
from src.core.response import ApiResponse, PageResponse
from src.core.state import (
    AgentState,
    DocsTeamState,
    KnowledgeTeamState,
    ResearchTeamState,
    TechnicalTeamState,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TEST_THREAD_ID = "thread-001"
TEST_USER_ID = "user-abc"
TEST_CONTENT = "Hello, agent!"


# ---------------------------------------------------------------------------
# TestAgentState
# ---------------------------------------------------------------------------
class TestAgentState:
    def test_minimal_state_construction(self):
        # Arrange / Act
        state: AgentState = {
            "thread_id": TEST_THREAD_ID,
            "user_id": TEST_USER_ID,
            "messages": [],
            "intent": None,
            "plan": None,
            "current_team": None,
            "team_output": None,
            "hitl_required": False,
            "final_response": None,
        }

        # Assert
        assert state["thread_id"] == TEST_THREAD_ID
        assert state["user_id"] == TEST_USER_ID
        assert state["hitl_required"] is False

    def test_add_messages_reducer_appends(self):
        # Arrange
        from langgraph.graph.message import add_messages

        existing = [HumanMessage(content="first")]
        new_msg = [HumanMessage(content="second")]

        # Act
        result = add_messages(existing, new_msg)

        # Assert
        assert len(result) == 2
        assert result[1].content == "second"

    def test_state_with_plan_and_intent(self):
        # Arrange / Act
        state: AgentState = {
            "thread_id": TEST_THREAD_ID,
            "user_id": TEST_USER_ID,
            "messages": [HumanMessage(content=TEST_CONTENT)],
            "intent": "research",
            "plan": ["step1", "step2"],
            "current_team": "research",
            "team_output": None,
            "hitl_required": False,
            "final_response": None,
        }

        # Assert
        assert state["intent"] == "research"
        assert state["plan"] == ["step1", "step2"]
        assert state["current_team"] == "research"


# ---------------------------------------------------------------------------
# TestTeamStates
# ---------------------------------------------------------------------------
class TestTeamStates:
    def test_docs_team_state_fields(self):
        # Arrange / Act
        state: DocsTeamState = {
            "messages": [],
            "raw_input": "meeting_audio.mp3",
            "script": None,
            "summary": None,
            "tasks": None,
            "sync_status": None,
        }

        # Assert
        assert state["raw_input"] == "meeting_audio.mp3"
        assert state["tasks"] is None

    def test_research_team_state_fields(self):
        state: ResearchTeamState = {
            "messages": [],
            "query": "SWOT analysis for fintech",
            "internal_knowledge": None,
            "web_results": None,
            "recommendation": None,
        }
        assert state["query"] == "SWOT analysis for fintech"

    def test_technical_team_state_verdict(self):
        state: TechnicalTeamState = {
            "messages": [],
            "design_spec": "button: primary, red",
            "code_output": "<button>Click</button>",
            "review_report": "Looks good",
            "verdict": "PASS",
        }
        assert state["verdict"] == "PASS"

    def test_knowledge_team_state_fields(self):
        state: KnowledgeTeamState = {
            "messages": [],
            "source_docs": ["https://docs.example.com"],
            "technical_summary": None,
            "lessons": None,
            "yaml_output": None,
        }
        assert state["source_docs"] == ["https://docs.example.com"]


# ---------------------------------------------------------------------------
# TestMessages
# ---------------------------------------------------------------------------
class TestMessages:
    def test_human_factory(self):
        msg = human(TEST_CONTENT)
        assert isinstance(msg, HumanMessage)
        assert msg.content == TEST_CONTENT

    def test_ai_factory(self):
        msg = ai("I am an AI")
        assert isinstance(msg, AIMessage)
        assert msg.content == "I am an AI"

    def test_system_factory(self):
        msg = system("You are helpful.")
        assert isinstance(msg, SystemMessage)
        assert msg.content == "You are helpful."

    def test_tool_factory(self):
        msg = tool("result data", tool_call_id="call-123")
        assert isinstance(msg, ToolMessage)
        assert msg.content == "result data"
        assert msg.tool_call_id == "call-123"

    def test_get_last_human_message_returns_last(self):
        messages = [
            HumanMessage(content="first"),
            AIMessage(content="response"),
            HumanMessage(content="second"),
        ]
        result = get_last_human_message(messages)
        assert result is not None
        assert result.content == "second"

    def test_get_last_human_message_returns_none_when_absent(self):
        messages = [AIMessage(content="only ai")]
        result = get_last_human_message(messages)
        assert result is None

    def test_filter_by_type_keeps_only_matching(self):
        messages = [
            HumanMessage(content="h1"),
            AIMessage(content="a1"),
            HumanMessage(content="h2"),
        ]
        humans = filter_by_type(messages, HumanMessage)
        assert len(humans) == 2
        assert all(isinstance(m, HumanMessage) for m in humans)

    def test_filter_by_type_empty_result(self):
        messages = [AIMessage(content="only ai")]
        result = filter_by_type(messages, HumanMessage)
        assert result == []


# ---------------------------------------------------------------------------
# TestExceptions
# ---------------------------------------------------------------------------
class TestExceptions:
    def test_agent_error_message_and_attrs(self):
        err = AgentError("something broke", agent_name="manager", code="ERR_01")
        assert str(err) == "something broke"
        assert err.agent_name == "manager"
        assert err.code == "ERR_01"

    def test_agent_error_is_exception(self):
        with pytest.raises(AgentError, match="something broke"):
            raise AgentError("something broke")

    def test_tool_error_inherits_agent_error(self):
        err = ToolError("tool failed", tool_name="web_search", agent_name="scout")
        assert isinstance(err, AgentError)
        assert err.tool_name == "web_search"
        assert err.agent_name == "scout"

    def test_quota_exceeded_error_message(self):
        err = QuotaExceededError(model="gemini-pro", usage=1000, limit=500)
        assert "gemini-pro" in str(err)
        assert err.usage == 1000
        assert err.limit == 500
        assert err.code == "QUOTA_EXCEEDED"

    def test_hitl_required_error(self):
        err = HitlRequiredException(reason="needs approval", thread_id="thread-99")
        assert "needs approval" in str(err)
        assert err.thread_id == "thread-99"
        assert err.code == "HITL_REQUIRED"

    def test_llm_provider_error(self):
        err = LLMProviderError("rate limited", provider="gemini")
        assert isinstance(err, AgentError)
        assert err.provider == "gemini"
        assert err.code == "LLM_PROVIDER_ERROR"

    def test_tool_error_is_catchable_as_agent_error(self):
        with pytest.raises(AgentError):
            raise ToolError("tool broke")


# ---------------------------------------------------------------------------
# TestApiResponse
# ---------------------------------------------------------------------------
class TestApiResponse:
    def test_ok_sets_success_true(self):
        resp = ApiResponse.ok(data={"key": "value"})
        assert resp.success is True
        assert resp.data == {"key": "value"}
        assert resp.message is None

    def test_ok_with_message(self):
        resp = ApiResponse.ok(data=42, message="Created")
        assert resp.message == "Created"
        assert resp.data == 42

    def test_error_sets_success_false(self):
        resp = ApiResponse.error("Not found")
        assert resp.success is False
        assert resp.data is None
        assert resp.message == "Not found"

    def test_model_dump_includes_all_fields(self):
        resp = ApiResponse.ok(data="hello")
        dumped = resp.model_dump()
        assert "success" in dumped
        assert "data" in dumped
        assert "message" in dumped

    def test_generic_with_list(self):
        resp: ApiResponse[list[int]] = ApiResponse.ok(data=[1, 2, 3])
        assert resp.data == [1, 2, 3]


# ---------------------------------------------------------------------------
# TestPageResponse
# ---------------------------------------------------------------------------
class TestPageResponse:
    def test_pages_computed_correctly(self):
        resp = PageResponse(items=["a", "b", "c"], total=25, page=0, size=10)
        assert resp.pages == 3

    def test_pages_exact_division(self):
        resp = PageResponse(items=[], total=20, page=1, size=10)
        assert resp.pages == 2

    def test_pages_zero_size_returns_zero(self):
        resp = PageResponse(items=[], total=10, page=0, size=0)
        assert resp.pages == 0

    def test_items_preserved(self):
        resp = PageResponse(items=[1, 2, 3], total=3, page=0, size=10)
        assert resp.items == [1, 2, 3]


# ---------------------------------------------------------------------------
# TestGuardrailNode
# ---------------------------------------------------------------------------
class TestGuardrailNode:
    def _make_valid_state(self) -> AgentState:
        return {
            "thread_id": TEST_THREAD_ID,
            "user_id": TEST_USER_ID,
            "messages": [HumanMessage(content=TEST_CONTENT)],
            "intent": None,
            "plan": None,
            "current_team": None,
            "team_output": None,
            "hitl_required": False,
            "final_response": None,
        }

    def test_valid_state_passes_default_check(self):
        # Arrange
        node = GuardrailNode.default()
        state = self._make_valid_state()

        # Act
        result = node(state)

        # Assert
        assert result["final_response"] is None

    def test_empty_messages_fails_default_check(self):
        # Arrange
        node = GuardrailNode.default()
        state = self._make_valid_state()
        state["messages"] = []

        # Act
        result = node(state)

        # Assert
        assert result["final_response"] is not None
        assert "empty" in result["final_response"].lower()

    def test_missing_user_id_fails_default_check(self):
        # Arrange
        node = GuardrailNode.default()
        state = self._make_valid_state()
        state["user_id"] = ""

        # Act
        result = node(state)

        # Assert
        assert result["final_response"] is not None
        assert "user_id" in result["final_response"]

    def test_custom_check_can_block(self):
        # Arrange
        def always_fail(s):
            return "blocked by policy"

        node = GuardrailNode(checks=[always_fail])
        state = self._make_valid_state()

        # Act
        result = node(state)

        # Assert
        assert result["final_response"] == "blocked by policy"
        assert result["hitl_required"] is False

    def test_custom_check_passes_through(self):
        # Arrange
        node = GuardrailNode(checks=[lambda s: None])
        state = self._make_valid_state()

        # Act
        result = node(state)

        # Assert
        assert result["final_response"] is None

    def test_first_failing_check_short_circuits(self):
        # Arrange
        second_called = []

        def first_fail(s):
            return "first failed"

        def second_check(s):
            second_called.append(True)
            return None

        node = GuardrailNode(checks=[first_fail, second_check])
        state = self._make_valid_state()

        # Act
        node(state)

        # Assert
        assert second_called == []


# ---------------------------------------------------------------------------
# TestLLMCache
# ---------------------------------------------------------------------------
class TestLLMCache:
    def test_make_key_is_deterministic(self):
        cache = RedisLLMCache(redis_url="redis://localhost:6389/0")
        key1 = cache.make_key("hello world", "gemini-pro")
        key2 = cache.make_key("hello world", "gemini-pro")
        assert key1 == key2

    def test_make_key_differs_by_model(self):
        cache = RedisLLMCache(redis_url="redis://localhost:6389/0")
        key1 = cache.make_key("same prompt", "gemini-pro")
        key2 = cache.make_key("same prompt", "beeknoee")
        assert key1 != key2

    def test_make_key_differs_by_prompt(self):
        cache = RedisLLMCache(redis_url="redis://localhost:6389/0")
        key1 = cache.make_key("prompt A", "gemini-pro")
        key2 = cache.make_key("prompt B", "gemini-pro")
        assert key1 != key2

    @pytest.mark.asyncio
    async def test_redis_get_returns_value(self):
        # Arrange
        cache = RedisLLMCache(redis_url="redis://localhost:6389/0")
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value="cached response")
        cache._client = mock_client

        # Act
        result = await cache.get("some-key")

        # Assert
        assert result == "cached response"
        mock_client.get.assert_called_once_with("llm_cache:some-key")

    @pytest.mark.asyncio
    async def test_redis_set_stores_with_ttl(self):
        # Arrange
        cache = RedisLLMCache(redis_url="redis://localhost:6389/0")
        mock_client = AsyncMock()
        cache._client = mock_client

        # Act
        await cache.set("k", "v", ttl=600)

        # Assert
        mock_client.set.assert_called_once_with("llm_cache:k", "v", ex=600)

    @pytest.mark.asyncio
    async def test_redis_delete_removes_key(self):
        # Arrange
        cache = RedisLLMCache(redis_url="redis://localhost:6389/0")
        mock_client = AsyncMock()
        cache._client = mock_client

        # Act
        await cache.delete("target-key")

        # Assert
        mock_client.delete.assert_called_once_with("llm_cache:target-key")

    @pytest.mark.asyncio
    async def test_redis_clear_deletes_all_prefixed_keys(self):
        # Arrange
        cache = RedisLLMCache(redis_url="redis://localhost:6389/0")
        mock_client = AsyncMock()
        mock_client.keys = AsyncMock(return_value=["llm_cache:k1", "llm_cache:k2"])
        cache._client = mock_client

        # Act
        await cache.clear()

        # Assert
        mock_client.delete.assert_called_once_with("llm_cache:k1", "llm_cache:k2")


# ---------------------------------------------------------------------------
# TestConflictResolver
# ---------------------------------------------------------------------------
class TestConflictResolver:
    def test_voting_returns_majority(self):
        # Arrange
        resolver = ConflictResolver()

        # Act
        result = resolver.resolve_sync(["A", "B", "A", "C", "A"], ConflictStrategy.VOTING)

        # Assert
        assert result == "A"

    def test_voting_single_item(self):
        resolver = ConflictResolver()
        result = resolver.resolve_sync(["only"], ConflictStrategy.VOTING)
        assert result == "only"

    def test_weighted_merge_picks_highest_weight(self):
        # Arrange
        resolver = ConflictResolver()

        # Act
        result = resolver.resolve_sync(
            ["low", "medium", "high"],
            ConflictStrategy.WEIGHTED_MERGE,
            weights=[0.1, 0.3, 0.9],
        )

        # Assert
        assert result == "high"

    def test_weighted_merge_defaults_equal_weights_first_wins(self):
        resolver = ConflictResolver()
        result = resolver.resolve_sync(
            ["alpha", "beta"], ConflictStrategy.WEIGHTED_MERGE
        )
        assert result in ("alpha", "beta")

    def test_weighted_merge_mismatched_lengths_raises(self):
        resolver = ConflictResolver()
        with pytest.raises(ValueError, match="same length"):
            resolver.resolve_sync(
                ["a", "b"], ConflictStrategy.WEIGHTED_MERGE, weights=[0.5]
            )

    def test_resolve_sync_llm_synthesis_raises(self):
        resolver = ConflictResolver()
        with pytest.raises(ValueError, match="resolve_async"):
            resolver.resolve_sync(["a"], ConflictStrategy.LLM_SYNTHESIS)

    def test_empty_outputs_raises(self):
        resolver = ConflictResolver()
        with pytest.raises(ValueError, match="empty"):
            resolver.resolve_sync([], ConflictStrategy.VOTING)

    @pytest.mark.asyncio
    async def test_resolve_async_voting_delegates(self):
        # Arrange
        resolver = ConflictResolver()

        # Act
        result = await resolver.resolve_async(["X", "X", "Y"], ConflictStrategy.VOTING)

        # Assert
        assert result == "X"

    @pytest.mark.asyncio
    async def test_llm_synthesis_calls_model(self):
        # Arrange
        resolver = ConflictResolver()
        mock_model = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "synthesized output"
        mock_model.ainvoke = AsyncMock(return_value=mock_response)

        # Act
        result = await resolver.resolve_async(
            ["output A", "output B"],
            ConflictStrategy.LLM_SYNTHESIS,
            model=mock_model,
        )

        # Assert
        assert result == "synthesized output"
        mock_model.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_synthesis_without_model_raises(self):
        resolver = ConflictResolver()
        with pytest.raises(ValueError, match="model is required"):
            await resolver.resolve_async(["a"], ConflictStrategy.LLM_SYNTHESIS, model=None)
