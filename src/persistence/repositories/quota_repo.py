from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.exceptions import QuotaExceededError
from src.persistence.models.quota import TokenUsage
from src.settings import settings


class QuotaRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def record_usage(self, user_id: str, model: str, tokens: int, period: str) -> TokenUsage:
        usage = TokenUsage(user_id=user_id, model=model, tokens=tokens, period=period)
        self._session.add(usage)
        await self._session.commit()
        await self._session.refresh(usage)
        return usage

    async def get_usage(self, user_id: str, period: str) -> list[TokenUsage]:
        result = await self._session.execute(
            select(TokenUsage)
            .where(TokenUsage.user_id == user_id, TokenUsage.period == period)
            .order_by(TokenUsage.recorded_at.desc())
        )
        return list(result.scalars().all())

    async def reset(self, user_id: str, period: str) -> None:
        await self._session.execute(
            delete(TokenUsage).where(TokenUsage.user_id == user_id, TokenUsage.period == period)
        )
        await self._session.commit()

    async def get_total_usage(self, user_id: str, period: str) -> int:
        usages = await self.get_usage(user_id, period)
        return sum(u.tokens for u in usages)

    async def check_limit(self, user_id: str, model: str, additional_tokens: int = 0) -> None:
        limits = {
            "daily": settings.quota_daily_token_limit,
            "weekly": settings.quota_weekly_token_limit,
            "monthly": settings.quota_monthly_token_limit,
        }
        for period, limit in limits.items():
            current = await self.get_total_usage(user_id, period)
            if current + additional_tokens > limit:
                raise QuotaExceededError(model=model, usage=current + additional_tokens, limit=limit)
