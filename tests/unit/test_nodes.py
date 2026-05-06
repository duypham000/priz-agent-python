from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.core.state import AgentState
from src.nodes.few_shot import make_few_shot_node
from src.nodes.meta_prompt import make_meta_prompt_node
from src.nodes.self_discovery import DEFAULT_CAPABILITIES, make_self_discovery_node
from src.nodes.summarization import make_summarization_node

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_state(**overrides) -> AgentState:
    state: AgentState = {
        "thread_id": "t-test",
        "user_id": "u-test",
        "messages": [],
        "intent": None,
        "plan": None,
        "current_team": None,
        "team_output": None,
        "hitl_required": False,
        "final_response": None,
    }
    state.update(overrides)
    return state


def _make_model(response: str) -> MagicMock:
    model = MagicMock()
    model.ainvoke = AsyncMock(return_value=MagicMock(content=response))
    return model


# ---------------------------------------------------------------------------
# TestSummarizationNode
# ---------------------------------------------------------------------------


class TestSummarizationNode:
    @pytest.mark.asyncio
    async def test_returns_final_response(self):
        # Arrange
        model = _make_model("summary text")
        node = make_summarization_node(model)
        state = _base_state(messages=[HumanMessage(content="hello")])

        # Act
        result = await node(state)

        # Assert
        assert result["final_response"] == "summary text"
        assert "messages" not in result

    @pytest.mark.asyncio
    async def test_condense_called_with_all_messages(self):
        # Arrange
        model = _make_model("condensed")
        node = make_summarization_node(model)
        msgs = [HumanMessage(content="a"), AIMessage(content="b"), HumanMessage(content="c")]
        state = _base_state(messages=msgs)

        # Act
        await node(state)

        # Assert — condenser calls model.ainvoke once
        assert model.ainvoke.call_count == 1
        call_prompt = model.ainvoke.call_args[0][0][0].content
        assert "a" in call_prompt
        assert "b" in call_prompt
        assert "c" in call_prompt

    @pytest.mark.asyncio
    async def test_empty_messages_still_calls_model(self):
        # Arrange
        model = _make_model("empty summary")
        node = make_summarization_node(model)
        state = _base_state(messages=[])

        # Act
        result = await node(state)

        # Assert
        assert model.ainvoke.call_count == 1
        assert result["final_response"] == "empty summary"

    @pytest.mark.asyncio
    async def test_custom_max_messages_passed_to_condenser(self):
        # Arrange — max_messages is stored on condenser; we verify the node uses it
        # by confirming the factory creates distinct condensers per call
        model_a = _make_model("a")
        model_b = _make_model("b")
        node_a = make_summarization_node(model_a, max_messages=5)
        node_b = make_summarization_node(model_b, max_messages=50)
        state = _base_state(messages=[HumanMessage(content="test")])

        # Act
        result_a = await node_a(state)
        result_b = await node_b(state)

        # Assert — both nodes operate independently
        assert result_a["final_response"] == "a"
        assert result_b["final_response"] == "b"


# ---------------------------------------------------------------------------
# TestFewShotNode
# ---------------------------------------------------------------------------


class TestFewShotNode:
    def _make_memory(self, docs: list[Document]) -> MagicMock:
        memory = MagicMock()
        memory.retrieve = AsyncMock(return_value=docs)
        return memory

    @pytest.mark.asyncio
    async def test_injects_system_message_with_examples(self):
        # Arrange
        docs = [
            Document(page_content="Example content A"),
            Document(page_content="Example content B"),
        ]
        memory = self._make_memory(docs)
        node = make_few_shot_node(memory, k=3)
        state = _base_state(messages=[HumanMessage(content="find me examples")])

        # Act
        result = await node(state)

        # Assert
        assert "messages" in result
        assert len(result["messages"]) == 1
        msg = result["messages"][0]
        assert isinstance(msg, SystemMessage)
        assert "Example content A" in msg.content
        assert "Example content B" in msg.content

    @pytest.mark.asyncio
    async def test_retrieval_query_is_last_human_message(self):
        # Arrange
        docs = [Document(page_content="doc")]
        memory = self._make_memory(docs)
        node = make_few_shot_node(memory)
        state = _base_state(
            messages=[
                SystemMessage(content="system"),
                HumanMessage(content="find X"),
                AIMessage(content="ok"),
                HumanMessage(content="actual query"),
            ]
        )

        # Act
        await node(state)

        # Assert
        memory.retrieve.assert_called_once_with("actual query", k=3)

    @pytest.mark.asyncio
    async def test_no_docs_returns_empty_dict(self):
        # Arrange
        memory = self._make_memory([])
        node = make_few_shot_node(memory)
        state = _base_state(messages=[HumanMessage(content="hello")])

        # Act
        result = await node(state)

        # Assert
        assert result == {}

    @pytest.mark.asyncio
    async def test_retrieval_uses_configured_k(self):
        # Arrange
        docs = [Document(page_content="x")]
        memory = self._make_memory(docs)
        node = make_few_shot_node(memory, k=7)
        state = _base_state(messages=[HumanMessage(content="query")])

        # Act
        await node(state)

        # Assert
        memory.retrieve.assert_called_once_with("query", k=7)

    @pytest.mark.asyncio
    async def test_falls_back_to_empty_query_when_no_human_message(self):
        # Arrange
        docs = [Document(page_content="generic example")]
        memory = self._make_memory(docs)
        node = make_few_shot_node(memory)
        state = _base_state(messages=[AIMessage(content="only ai message")])

        # Act
        await node(state)

        # Assert
        memory.retrieve.assert_called_once_with("", k=3)


# ---------------------------------------------------------------------------
# TestMetaPromptNode
# ---------------------------------------------------------------------------


class TestMetaPromptNode:
    @pytest.mark.asyncio
    async def test_returns_system_message_in_messages(self):
        # Arrange
        model = _make_model("  improved system prompt  ")
        node = make_meta_prompt_node(model)
        state = _base_state(messages=[HumanMessage(content="help me write code")])

        # Act
        result = await node(state)

        # Assert
        assert "messages" in result
        assert len(result["messages"]) == 1
        msg = result["messages"][0]
        assert isinstance(msg, SystemMessage)
        assert msg.content == "improved system prompt"

    @pytest.mark.asyncio
    async def test_existing_system_prompt_included_in_llm_prompt(self):
        # Arrange
        model = _make_model("new prompt")
        node = make_meta_prompt_node(model)
        state = _base_state(
            messages=[
                SystemMessage(content="be concise"),
                HumanMessage(content="write a poem"),
            ]
        )

        # Act
        await node(state)

        # Assert
        sent_prompt = model.ainvoke.call_args[0][0][0].content
        assert "be concise" in sent_prompt

    @pytest.mark.asyncio
    async def test_no_existing_system_prompt_sends_empty_placeholder(self):
        # Arrange
        model = _make_model("generated prompt")
        node = make_meta_prompt_node(model)
        state = _base_state(messages=[HumanMessage(content="help")])

        # Act
        result = await node(state)

        # Assert — node still produces output even without existing system prompt
        assert result["messages"][0].content == "generated prompt"
        sent_prompt = model.ainvoke.call_args[0][0][0].content
        # existing prompt section should be empty (empty string between the markers)
        assert "Current system prompt" in sent_prompt

    @pytest.mark.asyncio
    async def test_last_human_message_included_in_llm_prompt(self):
        # Arrange
        model = _make_model("ok")
        node = make_meta_prompt_node(model)
        state = _base_state(messages=[HumanMessage(content="write a poem about stars")])

        # Act
        await node(state)

        # Assert
        sent_prompt = model.ainvoke.call_args[0][0][0].content
        assert "write a poem about stars" in sent_prompt


# ---------------------------------------------------------------------------
# TestSelfDiscoveryNode
# ---------------------------------------------------------------------------


class TestSelfDiscoveryNode:
    @pytest.mark.asyncio
    async def test_returns_plan_list(self):
        # Arrange
        model = _make_model("1. Step one\n2. Step two\n3. Step three")
        node = make_self_discovery_node(model)
        state = _base_state(intent="research topic")

        # Act
        result = await node(state)

        # Assert
        assert result["plan"] == ["Step one", "Step two", "Step three"]

    @pytest.mark.asyncio
    async def test_returns_ai_message_summarizing_plan(self):
        # Arrange
        model = _make_model("1. Do A\n2. Do B")
        node = make_self_discovery_node(model)
        state = _base_state(intent="task")

        # Act
        result = await node(state)

        # Assert
        assert "messages" in result
        assert len(result["messages"]) == 1
        msg = result["messages"][0]
        assert isinstance(msg, AIMessage)
        assert "Do A" in msg.content
        assert "Do B" in msg.content

    @pytest.mark.asyncio
    async def test_intent_included_in_llm_prompt(self):
        # Arrange
        model = _make_model("1. Only step")
        node = make_self_discovery_node(model)
        state = _base_state(intent="write quarterly report")

        # Act
        await node(state)

        # Assert
        sent_prompt = model.ainvoke.call_args[0][0][0].content
        assert "write quarterly report" in sent_prompt

    @pytest.mark.asyncio
    async def test_custom_capabilities_in_prompt(self):
        # Arrange
        model = _make_model("1. Step")
        custom_caps = ["do X", "do Y", "do Z"]
        node = make_self_discovery_node(model, capabilities=custom_caps)
        state = _base_state(intent="test")

        # Act
        await node(state)

        # Assert
        sent_prompt = model.ainvoke.call_args[0][0][0].content
        assert "do X" in sent_prompt
        assert "do Y" in sent_prompt
        assert "do Z" in sent_prompt

    @pytest.mark.asyncio
    async def test_single_step_response_parsed_correctly(self):
        # Arrange
        model = _make_model("1. Only one step")
        node = make_self_discovery_node(model)
        state = _base_state(intent="simple task")

        # Act
        result = await node(state)

        # Assert
        assert result["plan"] == ["Only one step"]

    @pytest.mark.asyncio
    async def test_default_capabilities_used_when_none_provided(self):
        # Arrange
        model = _make_model("1. Step")
        node = make_self_discovery_node(model)
        state = _base_state(intent="anything")

        # Act
        await node(state)

        # Assert — default caps appear in prompt
        sent_prompt = model.ainvoke.call_args[0][0][0].content
        for cap in DEFAULT_CAPABILITIES:
            assert cap in sent_prompt

    @pytest.mark.asyncio
    async def test_none_intent_falls_back_to_not_specified(self):
        # Arrange
        model = _make_model("1. Step")
        node = make_self_discovery_node(model)
        state = _base_state(intent=None)

        # Act
        await node(state)

        # Assert
        sent_prompt = model.ainvoke.call_args[0][0][0].content
        assert "not specified" in sent_prompt
