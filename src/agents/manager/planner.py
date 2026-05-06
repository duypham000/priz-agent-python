import json
from typing import Callable

from langchain_core.language_models import BaseChatModel

from src.agents.manager.state import ManagerState
from src.core.messages import HumanMessage, SystemMessage, get_last_human_message

_PLANNER_SYSTEM_PROMPT = """Create a concise step-by-step execution plan for the user's request.
Respond with a JSON array of step strings. Provide 2-4 steps maximum.
Example: ["Step 1: Analyze input", "Step 2: Process data", "Step 3: Deliver output"]"""


def _parse_plan(text: str) -> list[str]:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(s) for s in parsed if s]
    except (json.JSONDecodeError, ValueError):
        pass
    lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]
    return lines if lines else ["Process the request"]


def make_planner_node(model: BaseChatModel) -> Callable:
    async def planner_node(state: ManagerState) -> dict:
        intent = state.get("intent") or "general"
        messages = state.get("messages", [])
        last_msg = get_last_human_message(messages)
        content = last_msg.content if last_msg else ""

        response = await model.ainvoke([
            SystemMessage(content=_PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=f"Intent: {intent}\nRequest: {content}"),
        ])

        return {"plan": _parse_plan(response.content)}

    return planner_node
