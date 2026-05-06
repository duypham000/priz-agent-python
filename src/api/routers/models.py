from fastapi import APIRouter, Depends

from src.api.deps import get_current_user, get_llm_registry
from src.core.auth import TokenUser
from src.core.response import ApiResponse
from src.llm.registry import LLMRegistry

router = APIRouter()


@router.get("")
async def list_models(
    current_user: TokenUser = Depends(get_current_user),
    llm_registry: LLMRegistry = Depends(get_llm_registry),
):
    model_list = llm_registry.list_models()
    default = model_list[0] if model_list else None
    return ApiResponse.ok({"models": model_list, "default": default})
