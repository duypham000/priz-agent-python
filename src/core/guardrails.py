from typing import Callable, Optional

from src.core.state import AgentState

GuardCheckFn = Callable[[AgentState], Optional[str]]


class GuardrailNode:
    """
    Wraps one or more safety-check functions into a LangGraph-compatible node.

    Each check returns None (pass) or a string error message (fail).
    On first failure the node short-circuits: sets `final_response` and marks
    `hitl_required = False` so the graph can route to an exit edge.
    """

    def __init__(self, checks: Optional[list[GuardCheckFn]] = None):
        self.checks: list[GuardCheckFn] = checks if checks is not None else [self._default_check]

    def __call__(self, state: AgentState) -> AgentState:
        for check in self.checks:
            error = check(state)
            if error:
                return {
                    **state,
                    "final_response": error,
                    "hitl_required": False,
                }
        return state

    @staticmethod
    def _default_check(state: AgentState) -> Optional[str]:
        if not state.get("user_id"):
            return "Guardrail failed: user_id is required"
        if not state.get("messages"):
            return "Guardrail failed: messages list is empty"
        return None

    @classmethod
    def default(cls) -> "GuardrailNode":
        return cls()
