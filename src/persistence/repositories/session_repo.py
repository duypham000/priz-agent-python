from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.persistence.models.session import Thread


class ThreadRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, user_id: str, title: str | None = None, agent_name: str = "summarizer") -> Thread:
        thread = Thread(user_id=user_id, title=title, agent_name=agent_name)
        self._session.add(thread)
        await self._session.commit()
        await self._session.refresh(thread)
        return thread

    async def get(self, thread_id: str) -> Thread | None:
        result = await self._session.execute(select(Thread).where(Thread.id == thread_id))
        return result.scalar_one_or_none()

    async def list_by_user(self, user_id: str, limit: int = 20, offset: int = 0) -> list[Thread]:
        result = await self._session.execute(
            select(Thread)
            .where(Thread.user_id == user_id)
            .order_by(Thread.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    async def update_status(self, thread_id: str, status: str) -> Thread | None:
        thread = await self.get(thread_id)
        if thread is None:
            return None
        thread.status = status
        await self._session.commit()
        await self._session.refresh(thread)
        return thread

    async def delete(self, thread_id: str) -> None:
        thread = await self.get(thread_id)
        if thread is not None:
            await self._session.delete(thread)
            await self._session.commit()
