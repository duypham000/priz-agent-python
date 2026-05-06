from typing_extensions import TypedDict

from src.core.state import DocsTeamState


class TaskItem(TypedDict):
    name: str
    deadline: str
    owner: str
    priority: str   # "high" | "medium" | "low"


__all__ = ["DocsTeamState", "TaskItem"]
