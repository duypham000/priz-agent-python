import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.memory.condensation import MemoryCondenser
from src.memory.entity_memory import EntityMemory
from src.memory.long_term import LongTermMemory, ScoredDoc
from src.memory.short_term import ShortTermMemory

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REDIS_URL = "redis://localhost:6389/0"
POSTGRES_URL = "postgresql+asyncpg://pagent:pagent@localhost:5442/pagent"
TEST_THREAD = "thread-001"
TEST_USER = "user-abc"


# ---------------------------------------------------------------------------
# TestShortTermMemory
# ---------------------------------------------------------------------------
class TestShortTermMemory:
    def _make_memory(self, mock_client: AsyncMock) -> ShortTermMemory:
        mem = ShortTermMemory(redis_url=REDIS_URL)
        mem._client = mock_client
        return mem

    @pytest.mark.asyncio
    async def test_add_message_pushes_and_sets_expire(self):
        # Arrange
        mock_client = AsyncMock()
        mem = self._make_memory(mock_client)
        msg = HumanMessage(content="hello")

        # Act
        await mem.add_message(TEST_THREAD, msg)

        # Assert
        assert mock_client.rpush.call_count == 1
        assert mock_client.expire.call_count == 1
        call_args = mock_client.rpush.call_args[0]
        assert call_args[0] == f"short_term:{TEST_THREAD}"
        serialized = json.loads(call_args[1])
        assert serialized["type"] == "human"

    @pytest.mark.asyncio
    async def test_get_recent_returns_last_n(self):
        # Arrange
        from langchain_core.messages import messages_from_dict, message_to_dict

        msg1 = HumanMessage(content="first")
        msg2 = AIMessage(content="second")
        raw = [
            json.dumps(message_to_dict(msg1)),
            json.dumps(message_to_dict(msg2)),
        ]
        mock_client = AsyncMock()
        mock_client.lrange = AsyncMock(return_value=raw)
        mem = self._make_memory(mock_client)

        # Act
        result = await mem.get_recent(TEST_THREAD, n=2)

        # Assert
        assert len(result) == 2
        assert isinstance(result[0], HumanMessage)
        assert isinstance(result[1], AIMessage)
        mock_client.lrange.assert_called_once_with(f"short_term:{TEST_THREAD}", -2, -1)

    @pytest.mark.asyncio
    async def test_clear_deletes_key(self):
        # Arrange
        mock_client = AsyncMock()
        mem = self._make_memory(mock_client)

        # Act
        await mem.clear(TEST_THREAD)

        # Assert
        mock_client.delete.assert_called_once_with(f"short_term:{TEST_THREAD}")

    @pytest.mark.asyncio
    async def test_get_recent_empty_list(self):
        # Arrange
        mock_client = AsyncMock()
        mock_client.lrange = AsyncMock(return_value=[])
        mem = self._make_memory(mock_client)

        # Act
        result = await mem.get_recent(TEST_THREAD, n=5)

        # Assert
        assert result == []

    @pytest.mark.asyncio
    async def test_add_message_uses_configured_ttl(self):
        # Arrange
        mock_client = AsyncMock()
        mem = ShortTermMemory(redis_url=REDIS_URL, ttl=3600)
        mem._client = mock_client

        # Act
        await mem.add_message(TEST_THREAD, HumanMessage(content="hi"))

        # Assert
        mock_client.expire.assert_called_once_with(f"short_term:{TEST_THREAD}", 3600)


# ---------------------------------------------------------------------------
# TestEntityMemory
# ---------------------------------------------------------------------------
class TestEntityMemory:
    def _make_memory(self, mock_client: AsyncMock) -> EntityMemory:
        mem = EntityMemory(redis_url=REDIS_URL)
        mem._client = mock_client
        return mem

    @pytest.mark.asyncio
    async def test_set_stores_json_in_hash(self):
        # Arrange
        mock_client = AsyncMock()
        mem = self._make_memory(mock_client)

        # Act
        await mem.set(TEST_USER, "name", "Alice")

        # Assert
        mock_client.hset.assert_called_once_with(
            f"entity:{TEST_USER}", "name", json.dumps("Alice")
        )
        mock_client.expire.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_returns_deserialized_value(self):
        # Arrange
        mock_client = AsyncMock()
        mock_client.hget = AsyncMock(return_value=json.dumps({"role": "admin"}))
        mem = self._make_memory(mock_client)

        # Act
        result = await mem.get(TEST_USER, "profile")

        # Assert
        assert result == {"role": "admin"}
        mock_client.hget.assert_called_once_with(f"entity:{TEST_USER}", "profile")

    @pytest.mark.asyncio
    async def test_get_nonexistent_returns_none(self):
        # Arrange
        mock_client = AsyncMock()
        mock_client.hget = AsyncMock(return_value=None)
        mem = self._make_memory(mock_client)

        # Act
        result = await mem.get(TEST_USER, "missing_key")

        # Assert
        assert result is None

    @pytest.mark.asyncio
    async def test_get_all_returns_all_fields(self):
        # Arrange
        mock_client = AsyncMock()
        mock_client.hgetall = AsyncMock(
            return_value={
                "name": json.dumps("Alice"),
                "age": json.dumps(30),
            }
        )
        mem = self._make_memory(mock_client)

        # Act
        result = await mem.get_all(TEST_USER)

        # Assert
        assert result == {"name": "Alice", "age": 30}
        mock_client.hgetall.assert_called_once_with(f"entity:{TEST_USER}")

    @pytest.mark.asyncio
    async def test_delete_removes_field(self):
        # Arrange
        mock_client = AsyncMock()
        mem = self._make_memory(mock_client)

        # Act
        await mem.delete(TEST_USER, "name")

        # Assert
        mock_client.hdel.assert_called_once_with(f"entity:{TEST_USER}", "name")

    @pytest.mark.asyncio
    async def test_set_complex_value_serialized(self):
        # Arrange
        mock_client = AsyncMock()
        mem = self._make_memory(mock_client)
        value = [1, 2, {"nested": True}]

        # Act
        await mem.set(TEST_USER, "data", value)

        # Assert
        mock_client.hset.assert_called_once_with(
            f"entity:{TEST_USER}", "data", json.dumps(value)
        )


# ---------------------------------------------------------------------------
# TestMemoryCondenser
# ---------------------------------------------------------------------------
class TestMemoryCondenser:
    def _make_model(self, response_content: str = "summary") -> AsyncMock:
        mock_model = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = response_content
        mock_model.ainvoke = AsyncMock(return_value=mock_response)
        return mock_model

    def test_should_condense_returns_false_below_threshold(self):
        # Arrange
        model = self._make_model()
        condenser = MemoryCondenser(model=model, max_messages=20)
        messages = [HumanMessage(content=f"msg {i}") for i in range(15)]

        # Act / Assert
        assert condenser.should_condense(messages) is False

    def test_should_condense_returns_true_above_threshold(self):
        # Arrange
        model = self._make_model()
        condenser = MemoryCondenser(model=model, max_messages=20)
        messages = [HumanMessage(content=f"msg {i}") for i in range(21)]

        # Act / Assert
        assert condenser.should_condense(messages) is True

    def test_should_condense_at_exact_threshold_returns_false(self):
        # Arrange
        model = self._make_model()
        condenser = MemoryCondenser(model=model, max_messages=10)
        messages = [HumanMessage(content=f"msg {i}") for i in range(10)]

        # Act / Assert
        assert condenser.should_condense(messages) is False

    @pytest.mark.asyncio
    async def test_condense_calls_model_with_formatted_history(self):
        # Arrange
        model = self._make_model("This is the summary")
        condenser = MemoryCondenser(model=model)
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there"),
        ]

        # Act
        result = await condenser.condense(messages)

        # Assert
        assert result == "This is the summary"
        model.ainvoke.assert_called_once()
        call_args = model.ainvoke.call_args[0][0]
        assert len(call_args) == 1
        assert "Hello" in call_args[0].content
        assert "Hi there" in call_args[0].content

    @pytest.mark.asyncio
    async def test_condense_empty_messages(self):
        # Arrange
        model = self._make_model("empty summary")
        condenser = MemoryCondenser(model=model)

        # Act
        result = await condenser.condense([])

        # Assert
        assert result == "empty summary"
        model.ainvoke.assert_called_once()


# ---------------------------------------------------------------------------
# TestLongTermMemory
# ---------------------------------------------------------------------------
class TestLongTermMemory:
    def _make_embeddings(self, vector: list[float] | None = None) -> AsyncMock:
        mock_emb = AsyncMock()
        mock_emb.aembed_query = AsyncMock(return_value=vector or [0.1] * 1536)
        return mock_emb

    def _make_model(self, content: str = "0.8") -> AsyncMock:
        mock_model = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = content
        mock_model.ainvoke = AsyncMock(return_value=mock_response)
        return mock_model

    @pytest.mark.asyncio
    async def test_embed_and_store_inserts_row(self):
        # Arrange
        mock_emb = self._make_embeddings()
        mem = LongTermMemory(postgres_url=POSTGRES_URL, embeddings=mock_emb)

        mock_conn = AsyncMock()
        mock_engine = AsyncMock()
        mock_engine.begin = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock(return_value=False)))
        mem._engine = mock_engine

        # Act
        await mem.embed_and_store("test content", {"source": "doc1"})

        # Assert
        mock_emb.aembed_query.assert_called_once_with("test content")
        mock_conn.execute.assert_called_once()
        call_params = mock_conn.execute.call_args[0][1]
        assert call_params["content"] == "test content"
        assert call_params["metadata"] == json.dumps({"source": "doc1"})

    @pytest.mark.asyncio
    async def test_retrieve_returns_documents(self):
        # Arrange
        mock_emb = self._make_embeddings()
        mem = LongTermMemory(postgres_url=POSTGRES_URL, embeddings=mock_emb)

        mock_result = MagicMock()
        mock_result.fetchall = MagicMock(
            return_value=[
                ("content A", json.dumps({"source": "a"})),
                ("content B", json.dumps({"source": "b"})),
            ]
        )
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value=mock_result)
        mock_engine = AsyncMock()
        mock_engine.connect = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock(return_value=False)))
        mem._engine = mock_engine

        # Act
        docs = await mem.retrieve("my query", k=2)

        # Assert
        assert len(docs) == 2
        assert docs[0].page_content == "content A"
        assert docs[0].metadata == {"source": "a"}
        assert docs[1].page_content == "content B"

    @pytest.mark.asyncio
    async def test_grade_relevance_filters_by_score(self):
        # Arrange
        mock_emb = self._make_embeddings()
        mem = LongTermMemory(postgres_url=POSTGRES_URL, embeddings=mock_emb)

        # First doc scores 0.9, second scores 0.2
        responses = [MagicMock(content="0.9"), MagicMock(content="0.2")]
        mock_model = AsyncMock()
        mock_model.ainvoke = AsyncMock(side_effect=responses)

        docs = [
            Document(page_content="relevant doc"),
            Document(page_content="irrelevant doc"),
        ]

        # Act
        scored = await mem.grade_relevance("query", docs, mock_model)

        # Assert
        assert len(scored) == 2
        assert scored[0].score == pytest.approx(0.9)
        assert scored[1].score == pytest.approx(0.2)
        passing = [s for s in scored if s.score >= 0.5]
        assert len(passing) == 1
        assert passing[0].document.page_content == "relevant doc"

    @pytest.mark.asyncio
    async def test_grade_relevance_handles_non_float_response(self):
        # Arrange
        mock_emb = self._make_embeddings()
        mem = LongTermMemory(postgres_url=POSTGRES_URL, embeddings=mock_emb)

        mock_model = AsyncMock()
        mock_model.ainvoke = AsyncMock(return_value=MagicMock(content="not a number"))

        docs = [Document(page_content="some doc")]

        # Act
        scored = await mem.grade_relevance("query", docs, mock_model)

        # Assert
        assert scored[0].score == 0.0

    @pytest.mark.asyncio
    async def test_rewrite_query_calls_model(self):
        # Arrange
        mock_emb = self._make_embeddings()
        mem = LongTermMemory(postgres_url=POSTGRES_URL, embeddings=mock_emb)
        mock_model = self._make_model("better query about AI")

        # Act
        result = await mem.rewrite_query("AI query", "too vague", mock_model)

        # Assert
        assert result == "better query about AI"
        mock_model.ainvoke.assert_called_once()
        prompt_content = mock_model.ainvoke.call_args[0][0][0].content
        assert "AI query" in prompt_content
        assert "too vague" in prompt_content
