from datetime import date

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.persistence.models.reminder import Reminder


class ReminderRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_pending(self, today: date) -> list[Reminder]:
        """Return all unsent reminders with deadline on or before today."""
        result = await self._session.execute(
            select(Reminder)
            .where(Reminder.deadline <= today, Reminder.is_sent.is_(False))
            .order_by(Reminder.deadline)
        )
        return list(result.scalars().all())

    async def mark_sent(self, reminder_id: str) -> None:
        result = await self._session.execute(
            select(Reminder).where(Reminder.id == reminder_id)
        )
        reminder = result.scalar_one_or_none()
        if reminder is not None:
            reminder.is_sent = True
            await self._session.commit()

    async def create(
        self,
        user_id: str,
        thread_id: str,
        task_name: str,
        deadline: date,
        owner: str = "Unassigned",
    ) -> Reminder:
        r = Reminder(
            user_id=user_id,
            thread_id=thread_id,
            task_name=task_name,
            deadline=deadline,
            owner=owner,
        )
        self._session.add(r)
        await self._session.commit()
        await self._session.refresh(r)
        return r
