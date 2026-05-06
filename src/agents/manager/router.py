from typing import Callable

from langchain_core.language_models import BaseChatModel

from src.agents.manager.state import ManagerState
from src.core.messages import HumanMessage, SystemMessage, get_last_human_message

_VALID_TEAMS = {"docs", "research", "technical", "knowledge"}

_INTENT_SYSTEM_PROMPT = """Classify the user's request into exactly one team name.
Respond with ONLY one word: docs, research, technical, or knowledge.

Teams:
- docs: meetings, transcription, task management, calendar, summaries
- research: analysis, market intelligence, strategy, knowledge retrieval, advisory
- technical: code, UI, design, implementation, software development
- knowledge: documentation, guidelines, learning, archiving, experience capture"""


def _parse_intent(text: str) -> str:
    normalized = text.strip().lower()
    if normalized in _VALID_TEAMS:
        return normalized
    for team in _VALID_TEAMS:
        if team in normalized:
            return team
    return "docs"


def make_intent_classifier_node(model: BaseChatModel) -> Callable:
    async def intent_classifier_node(state: ManagerState) -> dict:
        messages = state.get("messages", [])
        last_msg = get_last_human_message(messages)
        content = last_msg.content if last_msg else ""

        response = await model.ainvoke([
            SystemMessage(content=_INTENT_SYSTEM_PROMPT),
            HumanMessage(content=content),
        ])

        intent = _parse_intent(response.content)
        return {"intent": intent, "current_team": intent}

    return intent_classifier_node
