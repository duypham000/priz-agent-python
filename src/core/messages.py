from typing import Optional, Type, TypeVar
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

__all__ = [
    "AIMessage",
    "BaseMessage",
    "HumanMessage",
    "SystemMessage",
    "ToolMessage",
    "human",
    "ai",
    "system",
    "tool",
    "get_last_human_message",
    "filter_by_type",
]

M = TypeVar("M", bound=BaseMessage)


def human(content: str) -> HumanMessage:
    return HumanMessage(content=content)


def ai(content: str) -> AIMessage:
    return AIMessage(content=content)


def system(content: str) -> SystemMessage:
    return SystemMessage(content=content)


def tool(content: str, tool_call_id: str) -> ToolMessage:
    return ToolMessage(content=content, tool_call_id=tool_call_id)


def get_last_human_message(messages: list[BaseMessage]) -> Optional[HumanMessage]:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg
    return None


def filter_by_type(messages: list[BaseMessage], msg_type: Type[M]) -> list[M]:
    return [m for m in messages if isinstance(m, msg_type)]
