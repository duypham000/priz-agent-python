from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Coroutine

import yaml
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.pregel import Pregel as CompiledGraph

from src.agents.base import BaseAgent
from src.agents.teams.technical.state import TechnicalTeamState
from src.core.exceptions import ToolError
from src.tools.builtin.code_runner import run_code

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM_PROMPT = (
    "You are a strict code reviewer and QA engineer. Review the provided code against "
    "the design specification and execution results. "
    "Structure your report with: Code Quality, Spec Compliance, Execution Results, Issues Found. "
    "End your review with exactly 'VERDICT: PASS' or 'VERDICT: FAIL'."
)

_PYTHON_INDICATORS = ("def ", "import ", "print(", "class ", "if __name__", "while ", "for ", "elif ", "pass\n", "pass\r", "pass")


def _load_prompt() -> str:
    path = Path(__file__).parents[4] / "prompts" / "teams" / "technical" / "quality_gatekeeper.yaml"
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return data["system_prompt"]
    except Exception:
        logger.debug("quality_gatekeeper prompt not found, using default")
        return _DEFAULT_SYSTEM_PROMPT


def _is_python(code: str) -> bool:
    return any(indicator in code for indicator in _PYTHON_INDICATORS)


def _parse_verdict(review_text: str) -> str:
    """Extract PASS or FAIL verdict from review text."""
    upper = review_text.upper()
    if "VERDICT: PASS" in upper:
        return "PASS"
    if "VERDICT: FAIL" in upper:
        return "FAIL"
    # Fallback: presence of error keywords defaults to FAIL
    if any(word in upper for word in ("ERROR", "BUG", "FAIL", "BROKEN", "INCORRECT")):
        return "FAIL"
    return "PASS"


def make_quality_gatekeeper_node(
    model: BaseChatModel,
) -> Callable[[TechnicalTeamState], Coroutine[Any, Any, dict]]:
    """Return a LangGraph node that runs code and reviews it against the design spec.

    Reads: code_output, design_spec
    Writes: review_report, verdict
    """
    system_prompt = _load_prompt()

    async def quality_gatekeeper_node(state: TechnicalTeamState) -> dict:
        code = state.get("code_output") or ""
        design_spec = state.get("design_spec") or "No design spec available."

        if not code:
            return {
                "review_report": "No code provided for review.",
                "verdict": "FAIL",
            }

        # Attempt execution only for Python code
        execution_summary: str
        if _is_python(code):
            try:
                exec_result = await run_code(code)
                rc = exec_result["returncode"]
                stdout = exec_result["stdout"][:500] if exec_result["stdout"] else "(empty)"
                stderr = exec_result["stderr"][:500] if exec_result["stderr"] else "(none)"
                execution_summary = (
                    f"Exit code: {rc}\n"
                    f"stdout:\n{stdout}\n"
                    f"stderr:\n{stderr}"
                )
                if rc != 0:
                    # Execution failed — skip LLM review, verdict is FAIL immediately
                    report = (
                        f"## Execution Results\nCode exited with code {rc}.\n\n"
                        f"stderr:\n{stderr}\n\n"
                        "VERDICT: FAIL"
                    )
                    return {"review_report": report, "verdict": "FAIL"}
            except ToolError as exc:
                execution_summary = f"Execution failed: {exc}"
                report = f"## Execution Results\n{execution_summary}\n\nVERDICT: FAIL"
                return {"review_report": report, "verdict": "FAIL"}
        else:
            execution_summary = "Non-Python code — execution skipped, static review only."

        user_content = (
            f"## Design Specification\n{design_spec}\n\n"
            f"## Generated Code\n```\n{code}\n```\n\n"
            f"## Execution Results\n{execution_summary}\n\n"
            "Review the code and provide your structured report."
        )
        response = await model.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ])
        review_text = response.content
        verdict = _parse_verdict(review_text)
        return {"review_report": review_text, "verdict": verdict}

    return quality_gatekeeper_node


class QualityGatekeeperAgent(BaseAgent):
    name = "quality_gatekeeper"

    def build_graph(self) -> CompiledGraph:
        node = make_quality_gatekeeper_node(self.adapter.get_model())
        graph = StateGraph(TechnicalTeamState)
        graph.add_node("quality_gatekeeper", node)
        graph.add_edge(START, "quality_gatekeeper")
        graph.add_edge("quality_gatekeeper", END)
        return graph.compile()
