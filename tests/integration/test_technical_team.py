from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field, PrivateAttr

from src.agents.manager.graph import ManagerAgent
from src.agents.manager.state import ManagerState
from src.agents.teams.technical.code_architect import make_code_architect_node
from src.agents.teams.technical.quality_gatekeeper import make_quality_gatekeeper_node
from src.agents.teams.technical.state import TechnicalTeamState
from src.agents.teams.technical.supervisor import _build_technical_pipeline
from src.agents.teams.technical.visual_interpreter import make_visual_interpreter_node
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


def _make_technical_state(**overrides) -> TechnicalTeamState:
    state: TechnicalTeamState = {
        "messages": [],
        "design_spec": None,
        "code_output": None,
        "review_report": None,
        "verdict": None,
    }
    state.update(overrides)
    return state


def _make_manager_state(**overrides) -> ManagerState:
    state: ManagerState = {
        "thread_id": "t-technical-test",
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
# TestVisualInterpreter
# ---------------------------------------------------------------------------


class TestVisualInterpreter:
    @pytest.mark.asyncio
    async def test_run_withTextDescription_populatesDesignSpec(self):
        # Arrange
        spec_response = (
            "## Component Hierarchy\n- LoginForm\n  - EmailInput\n  - PasswordInput\n  - SubmitButton\n"
            "## Layout\nFlexbox column, centered\n"
            "## Colors & Typography\nPrimary: #4A90E2, Font: Inter 16px"
        )
        adapter = MockAdapter(responses=[spec_response])
        node = make_visual_interpreter_node(adapter.get_model())
        state = _make_technical_state(
            messages=[HumanMessage(content="A login form with email and password fields and a blue submit button.")]
        )

        # Act
        result = await node(state)

        # Assert
        assert result["design_spec"] is not None
        assert len(result["design_spec"]) > 0

    @pytest.mark.asyncio
    async def test_run_withNoMessages_returnsNoInputMessage(self):
        # Arrange
        adapter = MockAdapter(responses=["should not be called"])
        node = make_visual_interpreter_node(adapter.get_model())
        state = _make_technical_state(messages=[])

        # Act
        result = await node(state)

        # Assert
        assert result["design_spec"] == "No design input provided."

    @pytest.mark.asyncio
    async def test_run_withMultimodalMessage_extractsTextParts(self):
        # Arrange
        spec_response = "## Component Hierarchy\n- Dashboard\n  - Header\n  - Chart"
        adapter = MockAdapter(responses=[spec_response])
        node = make_visual_interpreter_node(adapter.get_model())
        multimodal_content = [
            {"type": "text", "text": "Analyze this dashboard design"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}},
        ]
        msg = HumanMessage(content=multimodal_content)
        state = _make_technical_state(messages=[msg])

        # Act
        result = await node(state)

        # Assert
        assert result["design_spec"] is not None
        assert len(result["design_spec"]) > 0


# ---------------------------------------------------------------------------
# TestCodeArchitect
# ---------------------------------------------------------------------------


class TestCodeArchitect:
    @pytest.mark.asyncio
    async def test_run_withDesignSpec_populatesCodeOutput(self):
        # Arrange
        code_response = "```python\nclass LoginForm:\n    def __init__(self):\n        self.email = ''\n        self.password = ''\n```"
        adapter = MockAdapter(responses=[code_response])
        node = make_code_architect_node(adapter.get_model())
        state = _make_technical_state(
            design_spec="## Component Hierarchy\n- LoginForm\n  - EmailInput\n  - PasswordInput"
        )

        # Act
        result = await node(state)

        # Assert
        assert result["code_output"] is not None
        # Markdown wrapper should be stripped
        assert "```" not in result["code_output"]
        assert "LoginForm" in result["code_output"]

    @pytest.mark.asyncio
    async def test_run_withNoSpec_stillProducesCode(self):
        # Arrange
        code_response = "```python\n# No spec provided\npass\n```"
        adapter = MockAdapter(responses=[code_response])
        node = make_code_architect_node(adapter.get_model())
        state = _make_technical_state(design_spec=None)

        # Act
        result = await node(state)

        # Assert
        assert result["code_output"] is not None
        assert "```" not in result["code_output"]

    @pytest.mark.asyncio
    async def test_run_withResponseWithoutCodeBlock_returnsFullResponse(self):
        # Arrange — LLM returns code without markdown fences
        raw_code = "def hello():\n    return 'world'"
        adapter = MockAdapter(responses=[raw_code])
        node = make_code_architect_node(adapter.get_model())
        state = _make_technical_state(design_spec="Simple hello function")

        # Act
        result = await node(state)

        # Assert
        assert result["code_output"] == raw_code


# ---------------------------------------------------------------------------
# TestQualityGatekeeper
# ---------------------------------------------------------------------------


class TestQualityGatekeeper:
    @pytest.mark.asyncio
    async def test_run_codePassesExecution_returnsPassVerdict(self):
        # Arrange
        review_response = "## Code Quality\nGood.\n## Spec Compliance\nFull.\n## Execution Results\nOK.\nVERDICT: PASS"
        adapter = MockAdapter(responses=[review_response])
        node = make_quality_gatekeeper_node(adapter.get_model())
        state = _make_technical_state(
            design_spec="Print hello world",
            code_output="print('hello world')",
        )

        # Act
        with patch(
            "src.agents.teams.technical.quality_gatekeeper.run_code",
            new=AsyncMock(return_value={"stdout": "hello world\n", "stderr": "", "returncode": 0}),
        ):
            result = await node(state)

        # Assert
        assert result["verdict"] == "PASS"
        assert result["review_report"] is not None

    @pytest.mark.asyncio
    async def test_run_codeFailsExecution_returnsFailVerdict(self):
        # Arrange
        adapter = MockAdapter(responses=["should not be called"])
        node = make_quality_gatekeeper_node(adapter.get_model())
        state = _make_technical_state(
            design_spec="Print hello",
            code_output="print(undefined_var)",
        )

        # Act
        with patch(
            "src.agents.teams.technical.quality_gatekeeper.run_code",
            new=AsyncMock(return_value={"stdout": "", "stderr": "NameError: name 'undefined_var' is not defined", "returncode": 1}),
        ):
            result = await node(state)

        # Assert
        assert result["verdict"] == "FAIL"
        assert "1" in result["review_report"]

    @pytest.mark.asyncio
    async def test_run_executionTimeout_returnsFailVerdict(self):
        # Arrange
        adapter = MockAdapter(responses=["should not be called"])
        node = make_quality_gatekeeper_node(adapter.get_model())
        state = _make_technical_state(
            design_spec="Infinite loop",
            code_output="while True: pass",
        )

        # Act
        with patch(
            "src.agents.teams.technical.quality_gatekeeper.run_code",
            new=AsyncMock(side_effect=ToolError("Code execution timed out after 10s", code="TIMEOUT")),
        ):
            result = await node(state)

        # Assert
        assert result["verdict"] == "FAIL"
        assert "failed" in result["review_report"].lower() or "timeout" in result["review_report"].lower()

    @pytest.mark.asyncio
    async def test_run_nonPythonCode_skipsExecutionReviewsOnly(self):
        # Arrange — TypeScript code: no Python indicators → skip run_code
        review_response = "## Code Quality\nClean TypeScript.\n## Spec Compliance\nMatches spec.\nVERDICT: PASS"
        adapter = MockAdapter(responses=[review_response])
        node = make_quality_gatekeeper_node(adapter.get_model())
        state = _make_technical_state(
            design_spec="Login button component",
            code_output="const LoginButton: React.FC = () => <button>Login</button>;",
        )

        run_code_mock = AsyncMock()

        # Act
        with patch("src.agents.teams.technical.quality_gatekeeper.run_code", new=run_code_mock):
            result = await node(state)

        # Assert
        run_code_mock.assert_not_awaited()
        assert result["verdict"] == "PASS"

    @pytest.mark.asyncio
    async def test_run_withNoCode_returnsFailVerdict(self):
        # Arrange
        adapter = MockAdapter(responses=["should not be called"])
        node = make_quality_gatekeeper_node(adapter.get_model())
        state = _make_technical_state(code_output=None)

        # Act
        result = await node(state)

        # Assert
        assert result["verdict"] == "FAIL"
        assert "No code provided" in result["review_report"]


# ---------------------------------------------------------------------------
# TestEndToEnd
# ---------------------------------------------------------------------------


class TestEndToEnd:
    @pytest.mark.asyncio
    async def test_fullPipeline_textSpec_codeGenAndReview_pass(self):
        # Arrange
        spec = "## Component Hierarchy\n- Calculator\n  - Display\n  - ButtonGrid\n## Layout\nGrid 4 cols"
        code = "class Calculator:\n    def add(self, a, b):\n        return a + b"
        review = "## Code Quality\nSimple, clean.\n## Spec Compliance\nBasic structure matches.\nVERDICT: PASS"

        adapter = _SequentialMockAdapter(responses=[spec, f"```python\n{code}\n```", review])
        pipeline = _build_technical_pipeline(adapter.get_model())

        initial_state: TechnicalTeamState = {
            "messages": [HumanMessage(content="Build a calculator with add and subtract operations")],
            "design_spec": None,
            "code_output": None,
            "review_report": None,
            "verdict": None,
        }

        # Act
        with patch(
            "src.agents.teams.technical.quality_gatekeeper.run_code",
            new=AsyncMock(return_value={"stdout": "", "stderr": "", "returncode": 0}),
        ):
            result = await pipeline.ainvoke(initial_state)

        # Assert
        assert result["design_spec"] is not None
        assert result["code_output"] is not None
        assert "```" not in result["code_output"]
        assert result["review_report"] is not None
        assert result["verdict"] == "PASS"

    @pytest.mark.asyncio
    async def test_fullPipeline_brokenCode_reviewFails(self):
        # Arrange
        spec = "## Component Hierarchy\n- Counter widget"
        broken_code = "def count(:\n    pass"  # syntax error
        review = "## Code Quality\nSyntax error.\nVERDICT: FAIL"

        adapter = _SequentialMockAdapter(responses=[spec, f"```python\n{broken_code}\n```", review])
        pipeline = _build_technical_pipeline(adapter.get_model())

        initial_state: TechnicalTeamState = {
            "messages": [HumanMessage(content="Build a counter widget")],
            "design_spec": None,
            "code_output": None,
            "review_report": None,
            "verdict": None,
        }

        # Act — execution fails with returncode=1 (syntax error)
        with patch(
            "src.agents.teams.technical.quality_gatekeeper.run_code",
            new=AsyncMock(return_value={"stdout": "", "stderr": "SyntaxError: invalid syntax", "returncode": 1}),
        ):
            result = await pipeline.ainvoke(initial_state)

        # Assert
        assert result["design_spec"] is not None
        assert result["code_output"] is not None
        assert result["verdict"] == "FAIL"


# ---------------------------------------------------------------------------
# TestManagerIntegration
# ---------------------------------------------------------------------------


class TestManagerIntegration:
    @pytest.mark.asyncio
    async def test_managerGraph_technicalRequest_executesTechnicalTeamAndReturnsOutput(self):
        # Arrange — responses for:
        # intent_classifier, planner,
        # visual_interpreter, code_architect,
        # (quality_gatekeeper run_code returns rc=0), quality_gatekeeper review,
        # validator
        spec = "## Component Hierarchy\n- TodoList\n  - TodoItem\n  - AddButton"
        code = "class TodoList:\n    def __init__(self):\n        self.items = []\n    def add(self, item):\n        self.items.append(item)"
        review = "## Code Quality\nClean, simple.\n## Spec Compliance\nMatches spec.\nVERDICT: PASS"

        adapter = _SequentialMockAdapter(responses=[
            "technical",                                          # intent_classifier
            '["Step 1: Interpret design", "Step 2: Generate code", "Step 3: Review"]',  # planner
            spec,                                                 # visual_interpreter
            f"```python\n{code}\n```",                           # code_architect
            review,                                              # quality_gatekeeper
            "SCORE: 0.92\nTechnical pipeline completed successfully.",  # validator
        ])
        agent = ManagerAgent(adapter=adapter)
        state = _make_manager_state(
            messages=[HumanMessage(content="Build a simple todo list component with add functionality")]
        )

        # Act
        with patch(
            "src.agents.teams.technical.quality_gatekeeper.run_code",
            new=AsyncMock(return_value={"stdout": "", "stderr": "", "returncode": 0}),
        ):
            result = await agent.run(state)

        # Assert
        assert result["current_team"] == "technical"
        assert result["team_output"] is not None
        assert result["final_response"] is not None
