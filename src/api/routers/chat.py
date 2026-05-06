from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from sqlalchemy.ext.asyncio import AsyncSession

from src.agents.registry import AgentRegistry
from src.api.deps import get_checkpointer, get_current_user, get_db, get_llm_registry
from src.api.dto.chat import ChatEvent, ChatRequest, HitlResumeRequest
from src.core.auth import TokenUser
from src.core.state import AgentState
from src.llm.registry import LLMRegistry
from src.llm.token_counter import count_tokens
from src.persistence.checkpointer import CheckpointerManager
from src.persistence.repositories.quota_repo import QuotaRepository
from src.persistence.repositories.session_repo import ThreadRepository

router = APIRouter()

_agent_registry = AgentRegistry()
_agent_registry.auto_discover()


async def _stream_agent(
    agent_name: str,
    state: AgentState,
    thread_id: str,
    thread_repo: ThreadRepository,
    llm_registry: LLMRegistry,
    user_id: str | None = None,
    quota_repo: QuotaRepository | None = None,
) -> AsyncIterator[str]:
    try:
        agent_class = _agent_registry.get_class(agent_name)
    except KeyError:
        yield f"data: {ChatEvent(type='error', message=f'Agent {agent_name!r} not found').model_dump_json()}\n\n"
        return

    adapter = llm_registry.get_with_fallback("default")
    agent = agent_class(adapter=adapter, config={"configurable": {"thread_id": thread_id}})

    try:
        await thread_repo.update_status(thread_id, "running")
        final_output = ""
        async for chunk in agent.stream(state):
            for node_name, node_output in chunk.items():
                if isinstance(node_output, dict) and node_output.get("hitl_required"):
                    reason = node_output.get("hitl_reason", "Human approval required")
                    event = ChatEvent(type="awaiting_approval", reason=reason, thread_id=thread_id)
                    yield f"data: {event.model_dump_json()}\n\n"
                    await thread_repo.update_status(thread_id, "awaiting_approval")
                    return
                out_str = str(node_output)
                final_output = out_str
                event = ChatEvent(type="node_complete", node=node_name, output=out_str)
                yield f"data: {event.model_dump_json()}\n\n"

        if user_id and quota_repo:
            output_tokens = count_tokens(final_output, adapter.provider_name)
            for period in ("daily", "weekly", "monthly"):
                await quota_repo.record_usage(user_id, adapter.model_name, output_tokens, period)

        yield f"data: {ChatEvent(type='done', final=final_output).model_dump_json()}\n\n"
        await thread_repo.update_status(thread_id, "completed")
    except Exception as exc:
        yield f"data: {ChatEvent(type='error', message=str(exc)).model_dump_json()}\n\n"
        await thread_repo.update_status(thread_id, "error")


@router.post("")
async def chat(
    request: ChatRequest,
    current_user: TokenUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    llm_registry: LLMRegistry = Depends(get_llm_registry),
):
    thread_repo = ThreadRepository(db)
    quota_repo = QuotaRepository(db)

    adapter = llm_registry.get_with_fallback("default")
    estimated_tokens = count_tokens(request.message or "", adapter.provider_name)
    await quota_repo.check_limit(current_user.id, adapter.model_name, additional_tokens=estimated_tokens)

    if request.thread_id:
        thread = await thread_repo.get(request.thread_id)
        if thread is None or thread.user_id != current_user.id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found")
        thread_id = request.thread_id
    else:
        title = request.message[:100] if request.message else None
        thread = await thread_repo.create(user_id=current_user.id, title=title, agent_name=request.agent_name)
        thread_id = thread.id

    state: AgentState = {
        "thread_id": thread_id,
        "user_id": current_user.id,
        "messages": [HumanMessage(content=request.message)],
        "intent": None,
        "plan": None,
        "current_team": None,
        "team_output": None,
        "hitl_required": False,
        "final_response": None,
    }

    return StreamingResponse(
        _stream_agent(
            request.agent_name,
            state,
            thread_id,
            thread_repo,
            llm_registry,
            user_id=current_user.id,
            quota_repo=quota_repo,
        ),
        media_type="text/event-stream",
        headers={"X-Thread-Id": thread_id},
    )


@router.post("/{thread_id}/resume")
async def resume_chat(
    thread_id: str,
    request: HitlResumeRequest,
    current_user: TokenUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    llm_registry: LLMRegistry = Depends(get_llm_registry),
    checkpointer: CheckpointerManager = Depends(get_checkpointer),  # noqa: ARG001
):
    thread_repo = ThreadRepository(db)
    quota_repo = QuotaRepository(db)
    thread = await thread_repo.get(thread_id)
    if thread is None or thread.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found")
    if thread.status != "awaiting_approval":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Thread is not awaiting approval"
        )

    feedback_content = request.feedback or ("Approved." if request.approved else "Rejected.")
    state: AgentState = {
        "thread_id": thread_id,
        "user_id": current_user.id,
        "messages": [HumanMessage(content=feedback_content)],
        "intent": None,
        "plan": None,
        "current_team": None,
        "team_output": None,
        "hitl_required": False,
        "final_response": None,
    }

    return StreamingResponse(
        _stream_agent(
            thread.agent_name,
            state,
            thread_id,
            thread_repo,
            llm_registry,
            user_id=current_user.id,
            quota_repo=quota_repo,
        ),
        media_type="text/event-stream",
        headers={"X-Thread-Id": thread_id},
    )
