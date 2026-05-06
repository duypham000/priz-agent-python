from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field, PrivateAttr

from src.agents.manager.graph import ManagerAgent
from src.agents.manager.state import ManagerState
from src.agents.teams.docs.data_processor import make_data_processor_node
from src.agents.teams.docs.state import DocsTeamState
from src.agents.teams.docs.summarizer import make_meeting_minutes_node
from src.agents.teams.docs.supervisor import _build_docs_pipeline
from src.agents.teams.docs.sync_manager import make_sync_manager_node
from src.agents.teams.docs.task_architect import make_task_architect_node
from src.core.exceptions import ToolError
from src.llm.base import BaseLLMAdapter
from src.llm.mock import MockAdapter
from src.llm.token_counter import TokenCountProvider, TokenCounter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _SequentialChatModel(BaseChatModel):
    """Fake chat model that returns responses in strict sequence (wraps around)."""

    responses: list[str] = Field(default_factory=list)
    _call_count: int = PrivateAttr(default=0)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,
        **kwargs,
    ) -> ChatResult:
        idx = self._call_count % len(self.responses)
        self._call_count += 1
        msg = AIMessage(content=self.responses[idx])
        return ChatResult(generations=[ChatGeneration(message=msg)])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,
        **kwargs,
    ) -> ChatResult:
        return self._generate(messages, stop, run_manager, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "sequential-fake"


class _SequentialMockAdapter(BaseLLMAdapter):
    """MockAdapter that returns responses in strict call order."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self._model: _SequentialChatModel | None = None

    def get_model(self) -> BaseChatModel:
        if self._model is None:
            self._model = _SequentialChatModel(responses=self.responses)
        return self._model

    def count_tokens(self, text: str) -> int:
        return TokenCounter().count(text, TokenCountProvider.MOCK)

    @property
    def provider_name(self) -> str:
        return "mock-sequential"

    @property
    def model_name(self) -> str:
        return "sequential-mock"


def _make_docs_state(**overrides) -> DocsTeamState:
    state: DocsTeamState = {
        "messages": [],
        "raw_input": "",
        "script": None,
        "summary": None,
        "tasks": None,
        "sync_status": None,
    }
    state.update(overrides)
    return state


def _make_manager_state(**overrides) -> ManagerState:
    state: ManagerState = {
        "thread_id": "t-docs-test",
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


# ---------------------------------------------------------------------------
# TestDataProcessor
# ---------------------------------------------------------------------------


class TestDataProcessor:
    @pytest.mark.asyncio
    async def test_run_withRawInput_populatesScript(self):
        # Arrange
        adapter = MockAdapter(responses=["## [00:01] Speaker A: Let's discuss Q3 goals."])
        node = make_data_processor_node(adapter.get_model())
        state = _make_docs_state(raw_input="Let's discuss Q3 goals.")

        # Act
        result = await node(state)

        # Assert
        assert result["script"] is not None
        assert "script" in result

    @pytest.mark.asyncio
    async def test_run_withEmptyInput_populatesScript(self):
        # Arrange
        adapter = MockAdapter(responses=["[No content provided]"])
        node = make_data_processor_node(adapter.get_model())
        state = _make_docs_state(raw_input="")

        # Act
        result = await node(state)

        # Assert
        assert result["script"] is not None


# ---------------------------------------------------------------------------
# TestSummarizer
# ---------------------------------------------------------------------------


class TestSummarizer:
    @pytest.mark.asyncio
    async def test_run_withScript_populatesSummary(self):
        # Arrange
        adapter = MockAdapter(responses=["## Meeting Minutes\n**Date:** Unknown\n## Executive Summary\nTest."])
        node = make_meeting_minutes_node(adapter.get_model())
        state = _make_docs_state(script="[00:01] Alice: We need to finalize the budget.")

        # Act
        result = await node(state)

        # Assert
        assert result["summary"] is not None

    @pytest.mark.asyncio
    async def test_run_withNoScript_fallsBackToRawInput(self):
        # Arrange
        adapter = MockAdapter(responses=["## Meeting Minutes\n## Executive Summary\nFallback summary."])
        node = make_meeting_minutes_node(adapter.get_model())
        state = _make_docs_state(raw_input="Raw meeting notes here.", script=None)

        # Act
        result = await node(state)

        # Assert
        assert result["summary"] is not None


# ---------------------------------------------------------------------------
# TestTaskArchitect
# ---------------------------------------------------------------------------


class TestTaskArchitect:
    @pytest.mark.asyncio
    async def test_run_withSummary_returnsTasksList(self):
        # Arrange
        tasks_json = '[{"name":"Review Q3 plan","deadline":"2026-05-01","owner":"Alice","priority":"high"}]'
        adapter = MockAdapter(responses=[tasks_json])
        node = make_task_architect_node(adapter.get_model())
        state = _make_docs_state(summary="## Meeting Minutes\n## Key Decisions\n- Review Q3 plan by Alice")

        # Act
        result = await node(state)

        # Assert
        assert isinstance(result["tasks"], list)
        assert len(result["tasks"]) == 1
        assert result["tasks"][0]["name"] == "Review Q3 plan"
        assert result["tasks"][0]["deadline"] == "2026-05-01"
        assert result["tasks"][0]["owner"] == "Alice"
        assert result["tasks"][0]["priority"] == "high"

    @pytest.mark.asyncio
    async def test_run_withMalformedJson_returnsEmptyList(self):
        # Arrange
        adapter = MockAdapter(responses=["not valid json at all"])
        node = make_task_architect_node(adapter.get_model())
        state = _make_docs_state(summary="Summary text here")

        # Act
        result = await node(state)

        # Assert
        assert result["tasks"] == []

    @pytest.mark.asyncio
    async def test_run_withMarkdownFencedJson_returnsTasksList(self):
        # Arrange
        fenced = '```json\n[{"name":"Task B","deadline":"TBD","owner":"Bob","priority":"low"}]\n```'
        adapter = MockAdapter(responses=[fenced])
        node = make_task_architect_node(adapter.get_model())
        state = _make_docs_state(summary="Some summary")

        # Act
        result = await node(state)

        # Assert
        assert len(result["tasks"]) == 1
        assert result["tasks"][0]["name"] == "Task B"

    @pytest.mark.asyncio
    async def test_run_withMultipleTasks_normalizesFields(self):
        # Arrange
        tasks_json = '[{"name":"Task A"},{"name":"Task B","deadline":"2026-06-01","owner":"Carol","priority":"medium"}]'
        adapter = MockAdapter(responses=[tasks_json])
        node = make_task_architect_node(adapter.get_model())
        state = _make_docs_state(summary="Meeting with multiple tasks")

        # Act
        result = await node(state)

        # Assert
        assert len(result["tasks"]) == 2
        assert result["tasks"][0]["deadline"] == "TBD"
        assert result["tasks"][0]["owner"] == "Unassigned"
        assert result["tasks"][0]["priority"] == "medium"


# ---------------------------------------------------------------------------
# TestSyncManager
# ---------------------------------------------------------------------------


class TestSyncManager:
    @pytest.mark.asyncio
    async def test_run_withTasks_callsCreateTaskAndPopulatesSyncStatus(self):
        # Arrange
        tasks = [{"name": "Task A", "deadline": "2026-05-01", "owner": "Bob", "priority": "high"}]
        adapter = MockAdapter(responses=["## Sync Status Report\n- Total tasks: 1\n- Synced successfully: 1\n- Failed: 0"])
        node = make_sync_manager_node(adapter.get_model())
        state = _make_docs_state(tasks=tasks)
        mock_result = {"status": "synced", "id": "notion-123"}

        # Act
        with patch("src.agents.teams.docs.sync_manager.create_task", new=AsyncMock(return_value=mock_result)):
            result = await node(state)

        # Assert
        assert result["sync_status"] is not None

    @pytest.mark.asyncio
    async def test_run_withCalendarError_syncStatusStillPopulated(self):
        # Arrange
        tasks = [{"name": "Task B", "deadline": "TBD", "owner": "Charlie", "priority": "low"}]
        adapter = MockAdapter(responses=["## Sync Status Report\n- Total tasks: 1\n- Synced successfully: 0\n- Failed: 1"])
        node = make_sync_manager_node(adapter.get_model())
        state = _make_docs_state(tasks=tasks)

        # Act
        with patch(
            "src.agents.teams.docs.sync_manager.create_task",
            new=AsyncMock(side_effect=ToolError("API key not configured", code="NOTION_NOT_CONFIGURED")),
        ):
            result = await node(state)

        # Assert
        assert result["sync_status"] is not None

    @pytest.mark.asyncio
    async def test_run_withEmptyTasks_producesStatusWithZeroTasks(self):
        # Arrange
        adapter = MockAdapter(responses=["## Sync Status Report\n- Total tasks: 0\n- Synced successfully: 0\n- Failed: 0"])
        node = make_sync_manager_node(adapter.get_model())
        state = _make_docs_state(tasks=[])

        # Act
        with patch("src.agents.teams.docs.sync_manager.create_task", new=AsyncMock()) as mock_ct:
            result = await node(state)

        # Assert
        assert result["sync_status"] is not None
        mock_ct.assert_not_called()


# ---------------------------------------------------------------------------
# TestEndToEnd
# ---------------------------------------------------------------------------


class TestEndToEnd:
    @pytest.mark.asyncio
    async def test_fullPipeline_textInput_populatesAllFields(self):
        # Arrange
        script_response = "## [00:01] Alice: We need to update the roadmap by May 1st."
        minutes_response = "## Meeting Minutes\n## Key Decisions\n- Update roadmap\n## Executive Summary\nQ3 planning."
        tasks_response = '[{"name":"Update roadmap","deadline":"2026-05-01","owner":"Alice","priority":"high"}]'
        sync_response = "## Sync Status Report\n- Total tasks: 1\n- Synced successfully: 1\n- Failed: 0"

        adapter = _SequentialMockAdapter(responses=[
            script_response,
            minutes_response,
            tasks_response,
            sync_response,
        ])
        pipeline = _build_docs_pipeline(adapter.get_model())

        initial_state: DocsTeamState = {
            "messages": [HumanMessage(content="Summarize our Q3 planning session.")],
            "raw_input": "Alice: We need to update the roadmap by May 1st.",
            "script": None,
            "summary": None,
            "tasks": None,
            "sync_status": None,
        }

        # Act
        with patch(
            "src.agents.teams.docs.sync_manager.create_task",
            new=AsyncMock(return_value={"status": "synced", "id": "test-id"}),
        ):
            result = await pipeline.ainvoke(initial_state)

        # Assert
        assert result["script"] is not None
        assert result["summary"] is not None
        assert isinstance(result["tasks"], list)
        assert result["sync_status"] is not None

    @pytest.mark.asyncio
    async def test_managerGraph_docsRequest_executesDocsTeamAndReturnsOutput(self):
        # Arrange — responses for: intent_classifier, planner, data_processor,
        #            summarizer, task_architect, sync_manager, validator
        adapter = _SequentialMockAdapter(responses=[
            "docs",
            '["Step 1: Process input", "Step 2: Summarize", "Step 3: Extract tasks"]',
            "## [00:01] Alice: Discuss roadmap.",
            "## Meeting Minutes\n## Executive Summary\nMeeting about roadmap.",
            '[{"name":"Update roadmap","deadline":"2026-05-01","owner":"Alice","priority":"high"}]',
            "## Sync Status Report\n- Total: 1\n- Synced: 1",
            "SCORE: 0.9\nMeeting minutes and action items have been processed and synced.",
        ])
        agent = ManagerAgent(adapter=adapter)
        state = _make_manager_state(
            messages=[HumanMessage(content="Process this meeting recording about Q3 roadmap")]
        )

        # Act
        with patch(
            "src.agents.teams.docs.sync_manager.create_task",
            new=AsyncMock(return_value={"status": "synced"}),
        ):
            result = await agent.run(state)

        # Assert
        assert result["current_team"] == "docs"
        assert result["team_output"] is not None
        assert result["final_response"] is not None
