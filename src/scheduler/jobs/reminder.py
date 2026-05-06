"""
Reminder job — Phase 15.

Reads pending reminders from DB → pushes notification (log or webhook).
"""
from __future__ import annotations

import logging
from datetime import date
from dataclasses import dataclass

import httpx

from src.persistence.database import AsyncSessionLocal
from src.persistence.repositories.reminder_repo import ReminderRepository
from src.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class ReminderRecord:
    id: str
    task_name: str
    deadline: str
    owner: str
    thread_id: str


async def fetch_pending_reminders(today: date | None = None) -> list[ReminderRecord]:
    """Fetch tasks due today or overdue from the DB."""
    today = today or date.today()
    async with AsyncSessionLocal() as session:
        repo = ReminderRepository(session)
        rows = await repo.get_pending(today)
        return [
            ReminderRecord(
                id=r.id,
                task_name=r.task_name,
                deadline=str(r.deadline),
                owner=r.owner,
                thread_id=r.thread_id,
            )
            for r in rows
        ]


async def send_reminder(record: ReminderRecord) -> bool:
    """Push a reminder notification.

    notification_method="log"     → structured log only
    notification_method="webhook" → HTTP POST to notification_webhook_url
    """
    method = settings.notification_method.lower()

    if method == "webhook":
        url = settings.notification_webhook_url
        if not url:
            logger.warning("notification_method=webhook but no notification_webhook_url configured")
            return False
        payload = {
            "task": record.task_name,
            "deadline": record.deadline,
            "owner": record.owner,
            "thread_id": record.thread_id,
        }
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
            logger.info("Webhook reminder sent: task=%r owner=%s", record.task_name, record.owner)
            return True
        except Exception as exc:
            logger.error("Webhook reminder failed: task=%r error=%s", record.task_name, exc)
            return False

    logger.info(
        "REMINDER: task=%r deadline=%s owner=%s thread=%s",
        record.task_name,
        record.deadline,
        record.owner,
        record.thread_id,
    )
    return True


async def run_reminder_job() -> dict:
    """Main entry point called by APScheduler."""
    today = date.today()
    pending = await fetch_pending_reminders(today)
    sent = 0
    failed = 0

    async with AsyncSessionLocal() as session:
        repo = ReminderRepository(session)
        for record in pending:
            success = await send_reminder(record)
            if success:
                await repo.mark_sent(record.id)
                sent += 1
            else:
                failed += 1

    summary = {"date": str(today), "pending": len(pending), "sent": sent, "failed": failed}
    logger.info("Reminder job complete: %s", summary)
    return summary
