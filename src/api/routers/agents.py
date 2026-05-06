from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from sqlalchemy.ext.asyncio import AsyncSession

from src.agents.registry import AgentRegistry
from src.api.deps import get_current_user, get_db, get_llm_registry
from src.api.dto.chat import ChatRequest
from src.api.routers.chat import _stream_agent
from src.core.auth import TokenUser
from src.core.state import AgentState
from src.llm.registry import LLMRegistry
from src.llm.token_counter import count_tokens
from src.persistence.repositories.quota_repo import QuotaRepository
from src.persistence.repositories.session_repo import ThreadRepository
from src.core.response import ApiResponse

router = APIRouter()

_agent_registry = AgentRegistry()
_agent_registry.auto_discover()

@router.get("")
async def list_agents(current_user: TokenUser = Depends(get_current_user)):
    agents = _agent_registry.list_agents()
    # Format to match elec_base expectations if needed, but simple list of strings is a start
    # ElecBase expects Agent[]: { id, name, description }
    # For now, let's return name as both id and name for compatibility
    return ApiResponse.ok([{"id": name, "name": name, "description": f"Agent {name}"} for name in agents])

@router.post("/{agent_name}/chat")
async def chat_compat(
    agent_name: str,
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
        thread = await thread_repo.create(user_id=current_user.id, title=title, agent_name=agent_name)
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
            agent_name,
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
