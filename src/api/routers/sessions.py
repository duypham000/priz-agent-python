from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_checkpointer, get_current_user, get_db
from src.api.dto.session import CheckpointResponse, SessionResponse
from src.core.auth import TokenUser
from src.core.response import ApiResponse, PageResponse
from src.persistence.checkpointer import CheckpointerManager
from src.persistence.repositories.session_repo import ThreadRepository

router = APIRouter()


@router.get("", response_model=ApiResponse[PageResponse[SessionResponse]])
async def list_sessions(
    page: int = Query(0, ge=0),
    size: int = Query(20, ge=1, le=100),
    current_user: TokenUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    thread_repo = ThreadRepository(db)
    threads = await thread_repo.list_by_user(current_user.id, limit=size, offset=page * size)
    items = [SessionResponse.model_validate(t) for t in threads]
    return ApiResponse.ok(PageResponse(items=items, total=len(items), page=page, size=size))


@router.get("/{thread_id}", response_model=ApiResponse[SessionResponse])
async def get_session(
    thread_id: str,
    current_user: TokenUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    thread_repo = ThreadRepository(db)
    thread = await thread_repo.get(thread_id)
    if thread is None or thread.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found")
    return ApiResponse.ok(SessionResponse.model_validate(thread))


@router.delete("/{thread_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    thread_id: str,
    current_user: TokenUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    thread_repo = ThreadRepository(db)
    thread = await thread_repo.get(thread_id)
    if thread is None or thread.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found")
    await thread_repo.delete(thread_id)


@router.get("/{thread_id}/checkpoints", response_model=ApiResponse[list[CheckpointResponse]])
async def list_checkpoints(
    thread_id: str,
    current_user: TokenUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    checkpointer: CheckpointerManager = Depends(get_checkpointer),
):
    thread_repo = ThreadRepository(db)
    thread = await thread_repo.get(thread_id)
    if thread is None or thread.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found")

    checkpoint_tuples = await checkpointer.list_checkpoints(thread_id)
    items = [
        CheckpointResponse(
            checkpoint_id=c.config["configurable"].get("checkpoint_id", ""),
            thread_id=thread_id,
            created_at=c.metadata.get("created_at") if c.metadata else None,
            metadata=c.metadata or {},
        )
        for c in checkpoint_tuples
    ]
    return ApiResponse.ok(items)


@router.get("/{thread_id}/messages")
async def get_messages(
    thread_id: str,
    current_user: TokenUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    checkpointer: CheckpointerManager = Depends(get_checkpointer),
):
    thread_repo = ThreadRepository(db)
    thread = await thread_repo.get(thread_id)
    if thread is None or thread.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found")

    config = {"configurable": {"thread_id": thread_id}}
    state = await checkpointer.saver.aget(config)
    
    if not state or "channel_values" not in state:
        return ApiResponse.ok([])

    messages = state["channel_values"].get("messages", [])
    
    serialized_messages = []
    for msg in messages:
        serialized_messages.append({
            "role": "user" if msg.type == "human" else "assistant",
            "content": msg.content,
            "type": msg.type,
            "metadata": getattr(msg, "response_metadata", {})
        })
    
    return ApiResponse.ok(serialized_messages)

