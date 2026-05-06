from typing import Literal

from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    agent_name: str = "summarizer"
    thread_id: str | None = None


class ChatEvent(BaseModel):
    type: Literal["token", "node_complete", "awaiting_approval", "done", "error"]
    content: str | None = None
    node: str | None = None
    output: str | None = None
    reason: str | None = None
    thread_id: str | None = None
    final: str | None = None
    message: str | None = None


class HitlResumeRequest(BaseModel):
    approved: bool
    feedback: str | None = None
