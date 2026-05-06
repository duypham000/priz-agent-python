from __future__ import annotations

from pathlib import Path

from src.core.exceptions import ToolError

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def read_guidelines(agent_name: str) -> str:
    """Read agent system prompt from prompts/{agent_name}/system.yaml."""
    path = _PROJECT_ROOT / "prompts" / agent_name / "system.yaml"
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ToolError(
            f"Guidelines not found for agent '{agent_name}' at {path}",
            tool_name="read_guidelines",
            code="GUIDELINES_NOT_FOUND",
        ) from exc
    except OSError as exc:
        raise ToolError(
            f"Failed to read guidelines for '{agent_name}': {exc}",
            tool_name="read_guidelines",
            code="GUIDELINES_READ_ERROR",
        ) from exc
