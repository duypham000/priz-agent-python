from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field, PrivateAttr

from src.agents.manager.graph import ManagerAgent
from src.agents.manager.state import ManagerState
from src.agents.teams.research.internal_brain import make_internal_brain_node
from src.agents.teams.research.market_scout import make_market_scout_node
from src.agents.teams.research.state import ResearchTeamState
from src.agents.teams.research.strategic_advisor import make_strategic_advisor_node
from src.agents.teams.research.supervisor import _build_research_pipeline
from src.core.exceptions import ToolError
from src.llm.base import BaseLLMAdapter
from src.llm.mock import MockAdapter
from src.llm.token_counter import TokenCountProvider, TokenCounter
from src.memory.long_term import LongTermMemory


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


def _make_research_state(**overrides) -> ResearchTeamState:
    state: ResearchTeamState = {
        "messages": [],
        "query": "",
        "internal_knowledge": None,
        "web_results": None,
        "recommendation": None,
    }
    state.update(overrides)
    return state


def _make_manager_state(**overrides) -> ManagerState:
    state: ManagerState = {
        "thread_id": "t-research-test",
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


def _make_mock_memory(docs: list[Document] | None = None) -> LongTermMemory:
    """Return a LongTermMemory mock with corrective_rag pre-configured."""
    memory = MagicMock(spec=LongTermMemory)
    memory.corrective_rag = AsyncMock(return_value=docs if docs is not None else [
        Document(page_content="Our Q3 revenue grew 15% YoY.", metadata={"source": "q3_report.md"}),
        Document(page_content="Market share in APAC increased to 23%.", metadata={"source": "market.md"}),
    ])
    return memory


# ---------------------------------------------------------------------------
# TestInternalBrain
# ---------------------------------------------------------------------------


class TestInternalBrain:
    @pytest.mark.asyncio
    async def test_run_withMemoryAndDocs_populatesInternalKnowledge(self):
        # Arrange
        adapter = MockAdapter(responses=["Based on the knowledge base: Q3 revenue grew 15%."])
        memory = _make_mock_memory()
        node = make_internal_brain_node(adapter.get_model(), memory)
        state = _make_research_state(query="What was our Q3 performance?")

        # Act
        result = await node(state)

        # Assert
        assert result["internal_knowledge"] is not None
        assert len(result["internal_knowledge"]) > 0
        memory.corrective_rag.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_run_withNoMemory_returnsNotConfiguredMessage(self):
        # Arrange
        adapter = MockAdapter(responses=["Should not be called"])
        node = make_internal_brain_node(adapter.get_model(), memory=None)
        state = _make_research_state(query="What is our market position?")

        # Act
        result = await node(state)

        # Assert
        assert result["internal_knowledge"] == "No internal knowledge base configured."

    @pytest.mark.asyncio
    async def test_run_withEmptyDocs_returnsNoKnowledgeMessage(self):
        # Arrange
        adapter = MockAdapter(responses=["Should not be called"])
        memory = _make_mock_memory(docs=[])
        node = make_internal_brain_node(adapter.get_model(), memory)
        state = _make_research_state(query="Unknown topic")

        # Act
        result = await node(state)

        # Assert
        assert "No relevant internal knowledge" in result["internal_knowledge"]

    @pytest.mark.asyncio
    async def test_run_withMemoryError_returnsUnavailableMessage(self):
        # Arrange
        adapter = MockAdapter(responses=["fallback"])
        memory = MagicMock(spec=LongTermMemory)
        memory.corrective_rag = AsyncMock(side_effect=Exception("DB connection failed"))
        node = make_internal_brain_node(adapter.get_model(), memory)
        state = _make_research_state(query="Any query")

        # Act
        result = await node(state)

        # Assert
        assert "unavailable" in result["internal_knowledge"].lower()


# ---------------------------------------------------------------------------
# TestMarketScout
# ---------------------------------------------------------------------------


class TestMarketScout:
    @pytest.mark.asyncio
    async def test_run_withValidQuery_populatesWebResults(self):
        # Arrange
        search_results = [
            {"title": "Market Report 2026", "url": "https://example.com/1", "content": "Market grew 20% in 2026."},
            {"title": "Competitor Analysis", "url": "https://example.com/2", "content": "Key competitors include X and Y."},
        ]
        followup_response = "latest market growth statistics 2026"
        synthesis_response = "## Key Findings\nMarket grew 20% with major competitors X and Y."

        adapter = _SequentialMockAdapter(responses=[followup_response, followup_response, synthesis_response])
        node = make_market_scout_node(adapter.get_model())
        state = _make_research_state(query="What is the current market growth?")

        # Act
        with patch("src.agents.teams.research.market_scout.web_search", new=AsyncMock(return_value=search_results)):
            result = await node(state)

        # Assert
        assert result["web_results"] is not None
        assert len(result["web_results"]) > 0

    @pytest.mark.asyncio
    async def test_run_latsExpansion_callsWebSearchMultipleTimes(self):
        # Arrange
        initial_results = [
            {"title": "Report A", "url": "https://a.com", "content": "Content A"},
            {"title": "Report B", "url": "https://b.com", "content": "Content B"},
        ]
        expansion_results = [{"title": "Deeper Report", "url": "https://c.com", "content": "Detailed content"}]

        call_count = 0

        async def mock_search(query, max_results=5):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return initial_results
            return expansion_results

        adapter = _SequentialMockAdapter(responses=["followup query 1", "followup query 2", "## Key Findings\nSynthesized."])
        node = make_market_scout_node(adapter.get_model())
        state = _make_research_state(query="competitive landscape AI market")

        # Act
        with patch("src.agents.teams.research.market_scout.web_search", new=mock_search):
            result = await node(state)

        # Assert
        assert result["web_results"] is not None
        assert call_count > 1, "LATS should trigger follow-up searches"

    @pytest.mark.asyncio
    async def test_run_withWebSearchError_returnsErrorMessage(self):
        # Arrange
        adapter = MockAdapter(responses=["synthesis"])
        node = make_market_scout_node(adapter.get_model())
        state = _make_research_state(query="any query")

        # Act
        with patch(
            "src.agents.teams.research.market_scout.web_search",
            new=AsyncMock(side_effect=ToolError("API key not configured", code="TAVILY_NOT_CONFIGURED")),
        ):
            result = await node(state)

        # Assert
        assert result["web_results"] is not None
        assert "unavailable" in result["web_results"].lower()


# ---------------------------------------------------------------------------
# TestStrategicAdvisor
# ---------------------------------------------------------------------------


class TestStrategicAdvisor:
    @pytest.mark.asyncio
    async def test_run_withBothSources_populatesRecommendation(self):
        # Arrange
        swot_response = (
            "## SWOT Analysis\n**Strengths:** Strong brand.\n**Weaknesses:** Limited reach.\n"
            "**Opportunities:** Growing market.\n**Threats:** Rising competition.\n"
            "## Strategic Recommendation\n### Priority Actions\n1. Expand distribution."
        )
        adapter = MockAdapter(responses=[swot_response])
        node = make_strategic_advisor_node(adapter.get_model())
        state = _make_research_state(
            query="Should we expand into APAC?",
            internal_knowledge="Our APAC revenue grew 15% last year.",
            web_results="## Key Findings\nAPAC market growing 30% annually.",
        )

        # Act
        result = await node(state)

        # Assert
        assert result["recommendation"] is not None
        assert len(result["recommendation"]) > 0

    @pytest.mark.asyncio
    async def test_run_withNoSources_stillProducesRecommendation(self):
        # Arrange
        adapter = MockAdapter(responses=["## SWOT Analysis\n**Strengths:** Unknown.\n## Strategic Recommendation\nInsufficient data."])
        node = make_strategic_advisor_node(adapter.get_model())
        state = _make_research_state(
            query="What is our strategy?",
            internal_knowledge=None,
            web_results=None,
        )

        # Act
        result = await node(state)

        # Assert
        assert result["recommendation"] is not None


# ---------------------------------------------------------------------------
# TestEndToEnd
# ---------------------------------------------------------------------------


class TestEndToEnd:
    @pytest.mark.asyncio
    async def test_fullPipeline_withMockMemoryAndSearch_populatesAllFields(self):
        # Arrange
        internal_synthesis = "Based on KB: Q3 revenue +15%, APAC market share 23%."
        # market scout: followup x2, then synthesis
        followup_q = "APAC market growth statistics 2026"
        web_synthesis = "## Key Findings\nAPAC growing 30%. Competitors are X and Y."
        swot = (
            "## SWOT Analysis\n**Strengths:** Revenue growth.\n"
            "## Strategic Recommendation\nExpand into APAC."
        )

        adapter = _SequentialMockAdapter(responses=[
            internal_synthesis,  # internal_brain synthesis call
            followup_q,          # market_scout LATS expansion for result 1
            followup_q,          # market_scout LATS expansion for result 2
            web_synthesis,       # market_scout final synthesis
            swot,                # strategic_advisor
        ])

        memory = _make_mock_memory()
        pipeline = _build_research_pipeline(adapter.get_model(), memory)

        initial_state: ResearchTeamState = {
            "messages": [HumanMessage(content="Should we expand into APAC?")],
            "query": "Should we expand into APAC?",
            "internal_knowledge": None,
            "web_results": None,
            "recommendation": None,
        }
        search_results = [
            {"title": "APAC Market 2026", "url": "https://x.com", "content": "APAC growing 30%."},
            {"title": "Competitor Report", "url": "https://y.com", "content": "Key players: X, Y."},
        ]

        # Act
        with patch(
            "src.agents.teams.research.market_scout.web_search",
            new=AsyncMock(return_value=search_results),
        ):
            result = await pipeline.ainvoke(initial_state)

        # Assert
        assert result["internal_knowledge"] is not None
        assert result["web_results"] is not None
        assert result["recommendation"] is not None

    @pytest.mark.asyncio
    async def test_fullPipeline_withoutMemory_skipsBrain(self):
        # Arrange — no memory: internal_brain returns fast, market_scout + advisor run
        followup_q = "latest AI market trends"
        web_synthesis = "## Key Findings\nAI market booming."
        swot = "## SWOT Analysis\n**Strengths:** Early mover.\n## Strategic Recommendation\nInvest in AI."

        adapter = _SequentialMockAdapter(responses=[
            followup_q,
            followup_q,
            web_synthesis,
            swot,
        ])
        pipeline = _build_research_pipeline(adapter.get_model(), memory=None)

        initial_state: ResearchTeamState = {
            "messages": [],
            "query": "AI market opportunities",
            "internal_knowledge": None,
            "web_results": None,
            "recommendation": None,
        }
        search_results = [{"title": "AI Report", "url": "https://ai.com", "content": "AI is booming."}]

        # Act
        with patch(
            "src.agents.teams.research.market_scout.web_search",
            new=AsyncMock(return_value=search_results),
        ):
            result = await pipeline.ainvoke(initial_state)

        # Assert
        assert result["internal_knowledge"] == "No internal knowledge base configured."
        assert result["web_results"] is not None
        assert result["recommendation"] is not None


# ---------------------------------------------------------------------------
# TestManagerIntegration
# ---------------------------------------------------------------------------


class TestManagerIntegration:
    @pytest.mark.asyncio
    async def test_managerGraph_researchRequest_executesResearchTeamAndReturnsOutput(self):
        # Arrange — responses for:
        # intent_classifier, planner,
        # (internal_brain has no memory → instant return),
        # market_scout expansion x2, market_scout synthesis,
        # strategic_advisor, validator
        adapter = _SequentialMockAdapter(responses=[
            "research",
            '["Step 1: Retrieve internal knowledge", "Step 2: Search web", "Step 3: Synthesize advisory"]',
            "deeper APAC statistics",       # market_scout LATS expansion for result 1
            "deeper APAC statistics",       # market_scout LATS expansion for result 2
            "## Key Findings\nAPAC market growing 30% YoY.",   # market_scout synthesis
            "## SWOT Analysis\n**Strengths:** Revenue.\n## Strategic Recommendation\nExpand APAC.",
            "SCORE: 0.88\nComprehensive SWOT analysis delivered.",
        ])
        agent = ManagerAgent(adapter=adapter)
        state = _make_manager_state(
            messages=[HumanMessage(content="Analyze our expansion opportunity into APAC market")]
        )
        search_results = [
            {"title": "APAC Growth Report", "url": "https://x.com", "content": "30% growth expected."},
            {"title": "Market Players", "url": "https://y.com", "content": "Key players analysis."},
        ]

        # Act — memory=None so internal_brain returns instantly; only web_search needs mocking
        with patch(
            "src.agents.teams.research.market_scout.web_search",
            new=AsyncMock(return_value=search_results),
        ):
            result = await agent.run(state)

        # Assert
        assert result["current_team"] == "research"
        assert result["team_output"] is not None
        assert result["final_response"] is not None
