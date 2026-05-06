from datetime import datetime

from pydantic import BaseModel


class SessionResponse(BaseModel):
    id: str
    user_id: str
    title: str | None
    status: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class CheckpointResponse(BaseModel):
    checkpoint_id: str
    thread_id: str
    created_at: datetime | None
    metadata: dict
