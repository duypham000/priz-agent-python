from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field, PrivateAttr

from src.agents.manager.graph import ManagerAgent
from src.agents.manager.state import ManagerState
from src.agents.teams.knowledge.experience_archivist import make_experience_archivist_node
from src.agents.teams.knowledge.knowledge_scout import make_knowledge_scout_node
from src.agents.teams.knowledge.state import KnowledgeTeamState
from src.agents.teams.knowledge.supervisor import _build_knowledge_pipeline
from src.agents.teams.knowledge.yaml_architect import make_yaml_architect_node
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


def _make_knowledge_state(**overrides) -> KnowledgeTeamState:
    state: KnowledgeTeamState = {
        "messages": [],
        "source_docs": None,
        "technical_summary": None,
        "lessons": None,
        "yaml_output": None,
    }
    state.update(overrides)
    return state


def _make_manager_state(**overrides) -> ManagerState:
    state: ManagerState = {
        "thread_id": "t-knowledge-test",
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


_SUMMARY_RESPONSE = (
    "## Overview\nLangGraph is a library for building stateful multi-agent applications.\n\n"
    "## Key Concepts\n- StateGraph\n- Nodes\n- Edges\n- Checkpointing\n\n"
    "## APIs & Interfaces\n`StateGraph(state_schema)` — creates a new graph.\n\n"
    "## Usage Patterns\n```python\ngraph = StateGraph(MyState)\ngraph.add_node('node1', fn)\n```\n\n"
    "## Important Constraints\nRequires Python 3.11+. Async-first design."
)

_LESSONS_RESPONSE = (
    "## Success Patterns\n- Structured state TypedDicts reduce bugs\n- Async nodes improve throughput\n\n"
    "## Failure Modes\n- Missing required state keys cause KeyError\n- Sync blocking calls in async nodes\n\n"
    "## Lessons Learned\n- Always initialize state with None defaults\n- Use run_in_executor for blocking I/O\n\n"
    "## Recommended Guidelines\n- ALWAYS use TypedDict for state schemas\n- NEVER block the event loop"
)

_YAML_RESPONSE = (
    "version: '1.0'\n"
    "generated_by: yaml_architect\n"
    "summary: Guidelines for LangGraph-based agent development\n"
    "best_practices:\n"
    "  - Use TypedDict for all state schemas\n"
    "  - Prefer async nodes over sync blocking calls\n"
    "  - Initialize all state fields with None defaults\n"
    "anti_patterns:\n"
    "  - Blocking the event loop inside agent nodes\n"
    "  - Hardcoding state field names as strings\n"
    "examples:\n"
    "  - scenario: Building a new agent node\n"
    "    guidance: Use make_*_node factory pattern returning an async callable\n"
)


# ---------------------------------------------------------------------------
# TestKnowledgeScout
# ---------------------------------------------------------------------------


class TestKnowledgeScout:
    @pytest.mark.asyncio
    async def test_run_withSourceDocs_populatesTechnicalSummary(self):
        # Arrange
        adapter = MockAdapter(responses=[_SUMMARY_RESPONSE])
        node = make_knowledge_scout_node(adapter.get_model())
        state = _make_knowledge_state(
            source_docs=["https://example.com/docs"],
            messages=[HumanMessage(content="Summarize LangGraph docs")],
        )

        # Act
        with patch(
            "src.agents.teams.knowledge.knowledge_scout._fetch_source",
            return_value="LangGraph is a library for building stateful agents.",
        ):
            result = await node(state)

        # Assert
        assert result["technical_summary"] is not None
        assert len(result["technical_summary"]) > 0

    @pytest.mark.asyncio
    async def test_run_withNoSourceDocs_fallsBackToLastMessage(self):
        # Arrange
        adapter = MockAdapter(responses=[_SUMMARY_RESPONSE])
        node = make_knowledge_scout_node(adapter.get_model())
        state = _make_knowledge_state(
            source_docs=None,
            messages=[HumanMessage(content="Analyze LangGraph architecture")],
        )

        # Act
        result = await node(state)

        # Assert
        assert result["technical_summary"] is not None
        assert len(result["technical_summary"]) > 0

    @pytest.mark.asyncio
    async def test_run_withEmptySourceDocs_fallsBackToMessages(self):
        # Arrange
        adapter = MockAdapter(responses=[_SUMMARY_RESPONSE])
        node = make_knowledge_scout_node(adapter.get_model())
        state = _make_knowledge_state(
            source_docs=[],
            messages=[HumanMessage(content="What is LangGraph?")],
        )

        # Act
        result = await node(state)

        # Assert
        assert result["technical_summary"] is not None

    @pytest.mark.asyncio
    async def test_run_withUnreachableUrl_returnsErrorInSummary(self):
        # Arrange
        adapter = MockAdapter(responses=[_SUMMARY_RESPONSE])
        node = make_knowledge_scout_node(adapter.get_model())
        state = _make_knowledge_state(
            source_docs=["https://nonexistent.invalid/docs"],
        )

        # Act — _fetch_source handles errors gracefully, LLM still runs
        result = await node(state)

        # Assert
        assert result["technical_summary"] is not None


# ---------------------------------------------------------------------------
# TestExperienceArchivist
# ---------------------------------------------------------------------------


class TestExperienceArchivist:
    @pytest.mark.asyncio
    async def test_run_withPhoenixAvailable_populatesLessons(self):
        # Arrange
        adapter = MockAdapter(responses=[_LESSONS_RESPONSE])
        node = make_experience_archivist_node(adapter.get_model())
        state = _make_knowledge_state(
            technical_summary=_SUMMARY_RESPONSE,
        )
        mock_spans = [
            {"name": "chat_model", "attributes": {"status": {"code": "OK"}, "input": {"value": "test"}, "output": {"value": "result"}}}
        ]

        # Act
        with patch(
            "src.agents.teams.knowledge.experience_archivist._fetch_phoenix_spans_sync",
            return_value=mock_spans,
        ):
            result = await node(state)

        # Assert
        assert result["lessons"] is not None
        assert len(result["lessons"]) > 0

    @pytest.mark.asyncio
    async def test_run_withPhoenixUnavailable_returnsLessons(self):
        # Arrange
        adapter = MockAdapter(responses=[_LESSONS_RESPONSE])
        node = make_experience_archivist_node(adapter.get_model())
        state = _make_knowledge_state(
            technical_summary=_SUMMARY_RESPONSE,
        )

        # Act — Phoenix is down, should fall back gracefully
        with patch(
            "src.agents.teams.knowledge.experience_archivist._fetch_phoenix_spans_sync",
            side_effect=ConnectionRefusedError("Phoenix offline"),
        ):
            result = await node(state)

        # Assert
        assert result["lessons"] is not None
        assert len(result["lessons"]) > 0

    @pytest.mark.asyncio
    async def test_run_withNoTechnicalSummary_stillProducesLessons(self):
        # Arrange
        adapter = MockAdapter(responses=[_LESSONS_RESPONSE])
        node = make_experience_archivist_node(adapter.get_model())
        state = _make_knowledge_state(technical_summary=None)

        # Act
        with patch(
            "src.agents.teams.knowledge.experience_archivist._fetch_phoenix_spans_sync",
            return_value=[],
        ):
            result = await node(state)

        # Assert
        assert result["lessons"] is not None


# ---------------------------------------------------------------------------
# TestYamlArchitect
# ---------------------------------------------------------------------------


class TestYamlArchitect:
    @pytest.mark.asyncio
    async def test_run_withValidSummaryAndLessons_generatesYaml(self):
        # Arrange
        adapter = MockAdapter(responses=[_YAML_RESPONSE])
        node = make_yaml_architect_node(adapter.get_model())
        state = _make_knowledge_state(
            technical_summary=_SUMMARY_RESPONSE,
            lessons=_LESSONS_RESPONSE,
        )

        # Act
        with patch("src.agents.teams.knowledge.yaml_architect.git_commit_files", new_callable=AsyncMock) as mock_git:
            mock_git.return_value = {"success": True, "stdout": "[main abc1234] chore: update", "stderr": ""}
            result = await node(state)

        # Assert
        assert result["yaml_output"] is not None
        assert len(result["yaml_output"]) > 0
        mock_git.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_withGitFailure_stillReturnsYaml(self):
        # Arrange — git commit fails but yaml_output should still be returned
        adapter = MockAdapter(responses=[_YAML_RESPONSE])
        node = make_yaml_architect_node(adapter.get_model())
        state = _make_knowledge_state(
            technical_summary=_SUMMARY_RESPONSE,
            lessons=_LESSONS_RESPONSE,
        )

        # Act
        with patch(
            "src.agents.teams.knowledge.yaml_architect.git_commit_files",
            new_callable=AsyncMock,
            side_effect=Exception("git not configured"),
        ):
            result = await node(state)

        # Assert — git failure is non-fatal
        assert result["yaml_output"] is not None

    @pytest.mark.asyncio
    async def test_run_withYamlInCodeFence_stripsMarkdown(self):
        # Arrange — LLM wraps YAML in code fences
        yaml_in_fence = f"```yaml\n{_YAML_RESPONSE}\n```"
        adapter = MockAdapter(responses=[yaml_in_fence])
        node = make_yaml_architect_node(adapter.get_model())
        state = _make_knowledge_state(
            technical_summary="summary",
            lessons="lessons",
        )

        # Act
        with patch("src.agents.teams.knowledge.yaml_architect.git_commit_files", new_callable=AsyncMock):
            result = await node(state)

        # Assert — fences stripped
        assert result["yaml_output"] is not None
        assert "```" not in result["yaml_output"]


# ---------------------------------------------------------------------------
# TestFullPipeline
# ---------------------------------------------------------------------------


class TestFullPipeline:
    @pytest.mark.asyncio
    async def test_fullPipeline_withSourceDocs_allFieldsPopulated(self):
        # Arrange — three sequential LLM calls
        adapter = _SequentialMockAdapter(responses=[
            _SUMMARY_RESPONSE,
            _LESSONS_RESPONSE,
            _YAML_RESPONSE,
        ])
        pipeline = _build_knowledge_pipeline(adapter.get_model())
        initial_state = _make_knowledge_state(
            source_docs=["https://example.com/docs"],
            messages=[HumanMessage(content="Learn from LangGraph docs")],
        )

        # Act
        with (
            patch(
                "src.agents.teams.knowledge.knowledge_scout._fetch_source",
                return_value="LangGraph docs content",
            ),
            patch(
                "src.agents.teams.knowledge.experience_archivist._fetch_phoenix_spans_sync",
                return_value=[],
            ),
            patch(
                "src.agents.teams.knowledge.yaml_architect.git_commit_files",
                new_callable=AsyncMock,
                return_value={"success": True, "stdout": "committed", "stderr": ""},
            ),
        ):
            result = await pipeline.ainvoke(initial_state)

        # Assert
        assert result["technical_summary"] is not None
        assert result["lessons"] is not None
        assert result["yaml_output"] is not None

    @pytest.mark.asyncio
    async def test_fullPipeline_withNoSourceDocs_completesSuccessfully(self):
        # Arrange
        adapter = _SequentialMockAdapter(responses=[
            _SUMMARY_RESPONSE,
            _LESSONS_RESPONSE,
            _YAML_RESPONSE,
        ])
        pipeline = _build_knowledge_pipeline(adapter.get_model())
        initial_state = _make_knowledge_state(
            messages=[HumanMessage(content="What are the best practices for LangGraph?")],
        )

        # Act
        with (
            patch(
                "src.agents.teams.knowledge.experience_archivist._fetch_phoenix_spans_sync",
                return_value=[],
            ),
            patch(
                "src.agents.teams.knowledge.yaml_architect.git_commit_files",
                new_callable=AsyncMock,
                return_value={"success": True, "stdout": "committed", "stderr": ""},
            ),
        ):
            result = await pipeline.ainvoke(initial_state)

        # Assert
        assert result["technical_summary"] is not None
        assert result["lessons"] is not None
        assert result["yaml_output"] is not None


# ---------------------------------------------------------------------------
# TestManagerIntegration
# ---------------------------------------------------------------------------


class TestManagerIntegration:
    @pytest.mark.asyncio
    async def test_managerIntegration_knowledgeTeam_returnsTeamOutput(self):
        # Arrange — manager needs responses for: guardrail, intent_classifier, planner, validator + 3 team nodes
        adapter = _SequentialMockAdapter(responses=[
            "safe",                  # guardrail
            "knowledge",             # intent_classifier
            "step1",                 # planner
            _SUMMARY_RESPONSE,       # knowledge_scout
            _LESSONS_RESPONSE,       # experience_archivist
            _YAML_RESPONSE,          # yaml_architect
            "Final answer based on knowledge team output.",  # validator
        ])
        agent = ManagerAgent(adapter=adapter)
        state = _make_manager_state(
            messages=[HumanMessage(content="Learn from LangGraph documentation")],
            current_team="knowledge",
        )

        # Act
        with (
            patch(
                "src.agents.teams.knowledge.experience_archivist._fetch_phoenix_spans_sync",
                return_value=[],
            ),
            patch(
                "src.agents.teams.knowledge.yaml_architect.git_commit_files",
                new_callable=AsyncMock,
                return_value={"success": True, "stdout": "committed", "stderr": ""},
            ),
        ):
            result = await agent.run(state)

        # Assert
        assert result.get("team_output") is not None
        team_output = result["team_output"]
        assert "Technical Summary" in team_output or len(team_output) > 0
