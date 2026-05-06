from typing import Optional

from src.core.state import AgentState


class ManagerState(AgentState, total=False):
    """AgentState extended with Manager-specific optional fields."""

    hitl_feedback: Optional[str]
    validation_score: Optional[float]
