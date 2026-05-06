"""
APScheduler setup — Phase 15.

Provides create_scheduler() which builds a configured AsyncIOScheduler.
Wired into FastAPI lifespan in main.py.
"""
from __future__ import annotations

import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from src.settings import settings

logger = logging.getLogger(__name__)


def create_scheduler() -> AsyncIOScheduler:
    """Build and configure the APScheduler instance.

    Jobs:
      - reminder_job        : IntervalTrigger every scheduler_reminder_interval_minutes
      - quota_reset_daily   : CronTrigger at midnight UTC (configurable)
      - quota_reset_weekly  : CronTrigger every Monday at midnight
    """
    from src.scheduler.jobs.reminder import run_reminder_job
    from src.scheduler.jobs.quota_reset import reset_daily, reset_weekly

    scheduler = AsyncIOScheduler(timezone=settings.scheduler_timezone)

    scheduler.add_job(
        run_reminder_job,
        trigger=IntervalTrigger(
            minutes=settings.scheduler_reminder_interval_minutes,
            timezone=settings.scheduler_timezone,
        ),
        id="reminder_job",
        name="Reminder Notification Job",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=300,
    )

    scheduler.add_job(
        reset_daily,
        trigger=CronTrigger(
            hour=settings.scheduler_quota_reset_daily_hour,
            minute=settings.scheduler_quota_reset_daily_minute,
            timezone=settings.scheduler_timezone,
        ),
        id="quota_reset_daily",
        name="Daily Quota Reset",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=3600,
    )

    scheduler.add_job(
        reset_weekly,
        trigger=CronTrigger(
            day_of_week=settings.scheduler_quota_reset_weekly_day,
            hour=settings.scheduler_quota_reset_daily_hour,
            minute=settings.scheduler_quota_reset_daily_minute,
            timezone=settings.scheduler_timezone,
        ),
        id="quota_reset_weekly",
        name="Weekly Quota Reset",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=3600,
    )

    return scheduler
