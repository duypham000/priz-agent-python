from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_current_user, get_db, get_user_client
from src.core.auth import TokenUser
from src.core.response import ApiResponse
from src.grpc.user_client import UserGrpcClient
from src.persistence.repositories.quota_repo import QuotaRepository
from src.settings import settings

router = APIRouter()

_DEFAULT_LIMITS = {
    "daily": settings.quota_daily_token_limit,
    "weekly": settings.quota_weekly_token_limit,
    "monthly": settings.quota_monthly_token_limit,
}


@router.get("")
async def get_quota(
    current_user: TokenUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    user_client: UserGrpcClient = Depends(get_user_client),
):
    # Fetch server-side quota config from base via gRPC; fall back to settings if unavailable
    daily_limit = _DEFAULT_LIMITS["daily"]
    ctx = await user_client.get_user_context(current_user.id)
    if ctx and ctx.quotas and ctx.quotas.max_tokens_per_day > 0:
        daily_limit = ctx.quotas.max_tokens_per_day

    limits = {
        "daily": daily_limit,
        "weekly": _DEFAULT_LIMITS["weekly"],
        "monthly": _DEFAULT_LIMITS["monthly"],
    }

    quota_repo = QuotaRepository(db)
    result = {}
    for period in ("daily", "weekly", "monthly"):
        usages = await quota_repo.get_usage(current_user.id, period)
        total = sum(u.tokens for u in usages)
        by_model: dict[str, int] = {}
        for u in usages:
            by_model[u.model] = by_model.get(u.model, 0) + u.tokens
        limit = limits[period]
        result[period] = {
            "total_tokens": total,
            "by_model": by_model,
            "limit": limit,
            "remaining": max(0, limit - total),
        }
    return ApiResponse.ok(result)
