import re
from typing import Callable

from langchain_core.language_models import BaseChatModel

from src.agents.manager.state import ManagerState
from src.core.messages import HumanMessage, SystemMessage

_VALIDATOR_SYSTEM_PROMPT = """Review the team's output and provide the final validated response for the user.
Start your response with a quality score on the first line in this exact format: SCORE: 0.95
Then provide the final polished response on subsequent lines."""

_SCORE_RE = re.compile(r"SCORE:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)


def _parse_score(text: str) -> float:
    match = _SCORE_RE.search(text)
    if match:
        try:
            return min(1.0, max(0.0, float(match.group(1))))
        except ValueError:
            pass
    return 0.0


def _parse_final_response(text: str, fallback: str) -> str:
    lines = text.strip().split("\n")
    content_lines = [ln for ln in lines if not _SCORE_RE.match(ln.strip())]
    result = "\n".join(content_lines).strip()
    return result if result else fallback


def make_validator_node(model: BaseChatModel) -> Callable:
    async def validator_node(state: ManagerState) -> dict:
        team_output = state.get("team_output") or ""

        response = await model.ainvoke([
            SystemMessage(content=_VALIDATOR_SYSTEM_PROMPT),
            HumanMessage(content=f"Team output:\n{team_output}"),
        ])

        score = _parse_score(response.content)
        final = _parse_final_response(response.content, team_output)
        return {"final_response": final, "validation_score": score}

    return validator_node
