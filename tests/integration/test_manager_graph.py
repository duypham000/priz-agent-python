from unittest.mock import patch

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from src.agents.manager.graph import ManagerAgent
from src.agents.manager.state import ManagerState
from src.agents.registry import AgentRegistry
from src.agents.teams.docs.state import DocsTeamState
from src.agents.teams.docs.summarizer import SummarizerAgent
from src.core.caching import LLMCache
from src.core.state import AgentState
from src.llm.mock import MockAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(**overrides) -> AgentState:
    state: AgentState = {
        "thread_id": "t-integration",
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


class _InMemoryCache(LLMCache):
    def __init__(self):
        self._store: dict = {}

    async def get(self, key: str):
        return self._store.get(key)

    async def set(self, key: str, value: str, ttl: int = 3600) -> None:
        self._store[key] = value

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)

    async def clear(self) -> None:
        self._store.clear()


# ---------------------------------------------------------------------------
# TestSummarizerAgent
# ---------------------------------------------------------------------------


class TestSummarizerAgent:
    @pytest.mark.asyncio
    async def test_run_withScript_returnsSummary(self):
        # Arrange
        adapter = MockAdapter(responses=["## Meeting Minutes\nExecutive summary here."])
        agent = SummarizerAgent(adapter=adapter)
        state = _make_docs_state(
            messages=[HumanMessage(content="Meeting transcript...")],
            script="[00:01] Alice: Let's discuss the roadmap.",
        )

        # Act
        result = await agent.run(state)

        # Assert
        assert result["summary"] is not None

    @pytest.mark.asyncio
    async def test_run_withRawInputFallback_returnsSummary(self):
        # Arrange
        adapter = MockAdapter(responses=["## Meeting Minutes\nNo script available."])
        agent = SummarizerAgent(adapter=adapter)
        state = _make_docs_state(raw_input="Some meeting notes.", script=None)

        # Act
        result = await agent.run(state)

        # Assert
        assert result["summary"] is not None

    @pytest.mark.asyncio
    async def test_stream_withScript_yieldsChunksWithSummary(self):
        # Arrange
        adapter = MockAdapter(responses=["## Meeting Minutes\nStreamed output."])
        agent = SummarizerAgent(adapter=adapter)
        state = _make_docs_state(script="[00:01] Speaker A: Hello.")

        # Act
        chunks = []
        async for chunk in agent.stream(state):
            chunks.append(chunk)

        # Assert
        assert len(chunks) > 0
        all_values = {}
        for chunk in chunks:
            for v in chunk.values():
                if isinstance(v, dict):
                    all_values.update(v)
        assert "summary" in all_values

    @pytest.mark.asyncio
    async def test_getGraph_calledTwice_returnsSameInstance(self):
        # Arrange
        adapter = MockAdapter(responses=["ok"])
        agent = SummarizerAgent(adapter=adapter)

        # Act
        graph1 = agent._get_graph()
        graph2 = agent._get_graph()

        # Assert
        assert graph1 is graph2


# ---------------------------------------------------------------------------
# TestAgentRegistry
# ---------------------------------------------------------------------------


class TestAgentRegistry:
    def test_register_validAgent_storesItByName(self):
        # Arrange
        registry = AgentRegistry()

        # Act
        registry.register(SummarizerAgent)

        # Assert
        assert "summarizer" in registry.list_agents()

    def test_getClass_registeredName_returnsClass(self):
        # Arrange
        registry = AgentRegistry()
        registry.register(SummarizerAgent)

        # Act
        cls = registry.get_class("summarizer")

        # Assert
        assert cls is SummarizerAgent

    def test_getClass_unknownName_raisesKeyError(self):
        # Arrange
        registry = AgentRegistry()

        # Act & Assert
        with pytest.raises(KeyError, match="not registered"):
            registry.get_class("unknown")

    def test_create_registeredName_returnsAgentInstance(self):
        # Arrange
        registry = AgentRegistry()
        registry.register(SummarizerAgent)
        adapter = MockAdapter(responses=["ok"])

        # Act
        agent = registry.create("summarizer", adapter=adapter)

        # Assert
        assert isinstance(agent, SummarizerAgent)
        assert agent.adapter is adapter

    def test_listAgents_afterRegister_containsName(self):
        # Arrange
        registry = AgentRegistry()

        # Act
        registry.register(SummarizerAgent)
        names = registry.list_agents()

        # Assert
        assert "summarizer" in names

    def test_autoDiscover_scansTeamsPackage_findsSummarizer(self):
        # Arrange
        registry = AgentRegistry()

        # Act
        registry.auto_discover("src.agents.teams")

        # Assert
        assert "summarizer" in registry.list_agents()


# ---------------------------------------------------------------------------
# TestBaseLLMAdapterCaching
# ---------------------------------------------------------------------------


class TestBaseLLMAdapterCaching:
    @pytest.mark.asyncio
    async def test_ainvoke_withoutCache_returnsModelResponse(self):
        # Arrange
        adapter = MockAdapter(responses=["response text"])

        # Act
        result = await adapter.ainvoke([HumanMessage(content="hello")])

        # Assert
        assert result.content == "response text"

    @pytest.mark.asyncio
    async def test_ainvoke_withCacheMiss_storesResultInCache(self):
        # Arrange
        adapter = MockAdapter(responses=["cached response"])
        cache = _InMemoryCache()
        msgs = [HumanMessage(content="hello")]

        # Act
        result = await adapter.ainvoke(msgs, cache=cache)

        # Assert — response returned and stored in cache
        assert result.content == "cached response"
        key = cache.make_key(str(msgs), adapter.model_name)
        stored = await cache.get(key)
        assert stored == "cached response"

    @pytest.mark.asyncio
    async def test_ainvoke_withCacheHit_skipsModelInvocation(self):
        # Arrange
        adapter = MockAdapter(responses=["model response"])
        cache = _InMemoryCache()
        msgs = [HumanMessage(content="hello")]
        key = cache.make_key(str(msgs), adapter.model_name)
        await cache.set(key, "from cache")

        # Act
        with patch.object(adapter, "get_model") as mock_get_model:
            result = await adapter.ainvoke(msgs, cache=cache)

        # Assert — model never called, cache value returned
        mock_get_model.assert_not_called()
        assert result.content == "from cache"


# ---------------------------------------------------------------------------
# Helpers for Manager tests
# ---------------------------------------------------------------------------


def _make_manager_state(**overrides) -> ManagerState:
    state: ManagerState = {
        "thread_id": "t-manager",
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
# TestManagerAgent
# ---------------------------------------------------------------------------


class TestManagerAgent:
    @pytest.mark.asyncio
    async def test_run_docsRequest_routesToDocsTeam(self):
        # Arrange
        adapter = MockAdapter(responses=["docs"])
        agent = ManagerAgent(adapter=adapter)
        state = _make_manager_state(messages=[HumanMessage(content="Summarize this meeting notes")])

        # Act
        result = await agent.run(state)

        # Assert
        assert result["current_team"] == "docs"
        assert result["final_response"] is not None

    @pytest.mark.asyncio
    async def test_run_researchRequest_routesToResearchTeam(self):
        # Arrange
        adapter = MockAdapter(responses=["research"])
        agent = ManagerAgent(adapter=adapter)
        state = _make_manager_state(messages=[HumanMessage(content="Analyze the market trends")])

        # Act
        result = await agent.run(state)

        # Assert
        assert result["current_team"] == "research"
        assert result["final_response"] is not None

    @pytest.mark.asyncio
    async def test_run_technicalRequest_routesToTechnicalTeam(self):
        # Arrange
        adapter = MockAdapter(responses=["technical"])
        agent = ManagerAgent(adapter=adapter)
        state = _make_manager_state(messages=[HumanMessage(content="Generate the UI component code")])

        # Act
        result = await agent.run(state)

        # Assert
        assert result["current_team"] == "technical"
        assert result["final_response"] is not None

    @pytest.mark.asyncio
    async def test_run_knowledgeRequest_routesToKnowledgeTeam(self):
        # Arrange
        adapter = MockAdapter(responses=["knowledge"])
        agent = ManagerAgent(adapter=adapter)
        state = _make_manager_state(messages=[HumanMessage(content="Archive this document into guidelines")])

        # Act
        result = await agent.run(state)

        # Assert
        assert result["current_team"] == "knowledge"
        assert result["final_response"] is not None

    @pytest.mark.asyncio
    async def test_run_guardrailFails_missingUserId_returnsError(self):
        # Arrange
        adapter = MockAdapter(responses=["docs"])
        agent = ManagerAgent(adapter=adapter)
        state = _make_manager_state(user_id="", messages=[HumanMessage(content="hello")])

        # Act
        result = await agent.run(state)

        # Assert
        assert result["final_response"] is not None
        assert "Guardrail failed" in result["final_response"]

    @pytest.mark.asyncio
    async def test_run_guardrailFails_emptyMessages_returnsError(self):
        # Arrange
        adapter = MockAdapter(responses=["docs"])
        agent = ManagerAgent(adapter=adapter)
        state = _make_manager_state(messages=[])

        # Act
        result = await agent.run(state)

        # Assert
        assert result["final_response"] is not None
        assert "Guardrail failed" in result["final_response"]

    @pytest.mark.asyncio
    async def test_run_withHitlRequired_graphInterrupts(self):
        # Arrange
        memory = MemorySaver()
        adapter = MockAdapter(responses=["docs"])
        agent = ManagerAgent(adapter=adapter, checkpointer=memory)
        state = _make_manager_state(
            thread_id="hitl-interrupt-1",
            messages=[HumanMessage(content="Approve this action")],
            hitl_required=True,
        )
        config = {"configurable": {"thread_id": "hitl-interrupt-1"}}

        # Act — first run must interrupt at HITL node
        result = await agent._get_graph().ainvoke(state, config=config)

        # Assert — graph signals interrupt
        assert "__interrupt__" in result

    @pytest.mark.asyncio
    async def test_run_afterHitlResume_completesWithFinalResponse(self):
        # Arrange
        memory = MemorySaver()
        adapter = MockAdapter(responses=["docs"])
        agent = ManagerAgent(adapter=adapter, checkpointer=memory)
        state = _make_manager_state(
            thread_id="hitl-resume-1",
            messages=[HumanMessage(content="Approve this plan")],
            hitl_required=True,
        )
        config = {"configurable": {"thread_id": "hitl-resume-1"}}

        # First run — interrupt
        await agent._get_graph().ainvoke(state, config=config)

        # Act — resume with human approval
        result = await agent._get_graph().ainvoke(Command(resume="approved"), config=config)

        # Assert
        assert result.get("final_response") is not None

    @pytest.mark.asyncio
    async def test_stream_managerGraph_yieldsNodeChunks(self):
        # Arrange
        adapter = MockAdapter(responses=["docs"])
        agent = ManagerAgent(adapter=adapter)
        state = _make_manager_state(messages=[HumanMessage(content="Summarize this meeting")])

        # Act
        chunks = []
        async for chunk in agent.stream(state):
            chunks.append(chunk)

        # Assert
        assert len(chunks) > 0
        node_names = set()
        for chunk in chunks:
            node_names.update(chunk.keys())
        assert len(node_names) > 0

    @pytest.mark.asyncio
    async def test_buildGraph_calledTwice_returnsSameInstance(self):
        # Arrange
        adapter = MockAdapter(responses=["docs"])
        agent = ManagerAgent(adapter=adapter)

        # Act
        graph1 = agent._get_graph()
        graph2 = agent._get_graph()

        # Assert
        assert graph1 is graph2

    def test_autoDiscover_scansManagerPackage_findsManager(self):
        # Arrange
        registry = AgentRegistry()

        # Act
        registry.auto_discover("src.agents.manager")

        # Assert
        assert "manager" in registry.list_agents()
