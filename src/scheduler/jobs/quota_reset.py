"""
Quota reset job — Phase 15.

reset_daily()  → called every midnight (UTC by default)
reset_weekly() → called every Monday midnight
"""
from __future__ import annotations

import logging

from sqlalchemy import select

from src.persistence.database import AsyncSessionLocal
from src.persistence.models.user import User
from src.persistence.repositories.quota_repo import QuotaRepository

logger = logging.getLogger(__name__)


async def _get_all_user_ids() -> list[str]:
    """Return IDs of all active users."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(User.id).where(User.is_active.is_(True))
        )
        return list(result.scalars().all())


async def reset_daily() -> dict:
    """Reset daily token counters for all active users."""
    user_ids = await _get_all_user_ids()
    async with AsyncSessionLocal() as session:
        repo = QuotaRepository(session)
        for user_id in user_ids:
            await repo.reset(user_id, "daily")
    summary = {"period": "daily", "users_reset": len(user_ids)}
    logger.info("Quota reset complete: %s", summary)
    return summary


async def reset_weekly() -> dict:
    """Reset weekly token counters for all active users."""
    user_ids = await _get_all_user_ids()
    async with AsyncSessionLocal() as session:
        repo = QuotaRepository(session)
        for user_id in user_ids:
            await repo.reset(user_id, "weekly")
    summary = {"period": "weekly", "users_reset": len(user_ids)}
    logger.info("Quota reset complete: %s", summary)
    return summary
