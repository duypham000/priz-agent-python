from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.persistence.models.reminder import Reminder
from src.persistence.repositories.reminder_repo import ReminderRepository
from src.scheduler.setup import create_scheduler

# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------

TEST_USER_ID = "user-001"
TEST_THREAD_ID = "thread-abc"
_NOW = datetime.now(timezone.utc)
_TODAY = date(2026, 4, 26)


def _make_reminder(**kwargs) -> MagicMock:
    defaults = dict(
        id="rem-001",
        user_id=TEST_USER_ID,
        thread_id=TEST_THREAD_ID,
        task_name="Write report",
        deadline=_TODAY,
        owner="Alice",
        is_sent=False,
        created_at=_NOW,
    )
    defaults.update(kwargs)
    r = MagicMock(spec=Reminder)
    for k, v in defaults.items():
        setattr(r, k, v)
    return r


# ---------------------------------------------------------------------------
# TestReminderRepository
# ---------------------------------------------------------------------------

class TestReminderRepository:
    def _make_repo(self) -> tuple[ReminderRepository, AsyncMock]:
        session = AsyncMock()
        return ReminderRepository(session), session

    @pytest.mark.asyncio
    async def test_getPending_returnsOnlyUnsentAndDue(self):
        # Arrange
        repo, session = self._make_repo()
        row = _make_reminder()
        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = [row]
        session.execute = AsyncMock(return_value=result_mock)

        # Act
        rows = await repo.get_pending(_TODAY)

        # Assert
        assert len(rows) == 1
        assert rows[0].task_name == "Write report"

    @pytest.mark.asyncio
    async def test_markSent_updatesIsSentAndCommits(self):
        # Arrange
        repo, session = self._make_repo()
        reminder = _make_reminder()
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = reminder
        session.execute = AsyncMock(return_value=result_mock)

        # Act
        await repo.mark_sent("rem-001")

        # Assert
        assert reminder.is_sent is True
        session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_markSent_nonExistentId_doesNotRaise(self):
        # Arrange
        repo, session = self._make_repo()
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = None
        session.execute = AsyncMock(return_value=result_mock)

        # Act & Assert (no exception)
        await repo.mark_sent("nonexistent")
        session.commit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_create_addsAndReturnsReminder(self):
        # Arrange
        repo, session = self._make_repo()
        session.refresh = AsyncMock()

        # Act
        r = await repo.create(
            user_id=TEST_USER_ID,
            thread_id=TEST_THREAD_ID,
            task_name="Send email",
            deadline=_TODAY,
            owner="Bob",
        )

        # Assert
        session.add.assert_called_once()
        session.commit.assert_awaited_once()


# ---------------------------------------------------------------------------
# TestReminderJob
# ---------------------------------------------------------------------------

class TestReminderJob:
    @pytest.mark.asyncio
    async def test_fetchPendingReminders_returnsReminderRecords(self):
        # Arrange
        row = _make_reminder()
        mock_repo = AsyncMock(spec=ReminderRepository)
        mock_repo.get_pending = AsyncMock(return_value=[row])

        with (
            patch("src.scheduler.jobs.reminder.AsyncSessionLocal") as mock_session_ctx,
            patch("src.scheduler.jobs.reminder.ReminderRepository", return_value=mock_repo),
        ):
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)
            mock_session_ctx.return_value = mock_session

            from src.scheduler.jobs.reminder import fetch_pending_reminders

            # Act
            records = await fetch_pending_reminders(_TODAY)

        # Assert
        assert len(records) == 1
        assert records[0].task_name == "Write report"
        assert records[0].id == "rem-001"
        assert records[0].owner == "Alice"

    @pytest.mark.asyncio
    async def test_sendReminder_logMethod_returnsTrue(self):
        # Arrange
        from src.scheduler.jobs.reminder import ReminderRecord, send_reminder

        record = ReminderRecord(
            id="rem-001", task_name="Task A", deadline="2026-04-26", owner="Alice", thread_id="t-1"
        )

        with patch("src.scheduler.jobs.reminder.settings") as mock_settings:
            mock_settings.notification_method = "log"

            # Act
            result = await send_reminder(record)

        # Assert
        assert result is True

    @pytest.mark.asyncio
    async def test_sendReminder_webhookMethod_postsAndReturnsTrue(self):
        # Arrange
        from src.scheduler.jobs.reminder import ReminderRecord, send_reminder

        record = ReminderRecord(
            id="rem-001", task_name="Task B", deadline="2026-04-26", owner="Bob", thread_id="t-2"
        )
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        with patch("src.scheduler.jobs.reminder.settings") as mock_settings:
            mock_settings.notification_method = "webhook"
            mock_settings.notification_webhook_url = "http://example.com/hook"

            with patch("src.scheduler.jobs.reminder.httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=False)
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_client_cls.return_value = mock_client

                # Act
                result = await send_reminder(record)

        # Assert
        assert result is True
        mock_client.post.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_sendReminder_webhookMethod_httpError_returnsFalse(self):
        # Arrange
        from src.scheduler.jobs.reminder import ReminderRecord, send_reminder
        import httpx

        record = ReminderRecord(
            id="rem-002", task_name="Task C", deadline="2026-04-26", owner="Carol", thread_id="t-3"
        )

        with patch("src.scheduler.jobs.reminder.settings") as mock_settings:
            mock_settings.notification_method = "webhook"
            mock_settings.notification_webhook_url = "http://example.com/hook"

            with patch("src.scheduler.jobs.reminder.httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=False)
                mock_client.post = AsyncMock(side_effect=Exception("connection refused"))
                mock_client_cls.return_value = mock_client

                # Act
                result = await send_reminder(record)

        # Assert
        assert result is False

    @pytest.mark.asyncio
    async def test_sendReminder_webhookMethod_noUrl_returnsFalse(self):
        # Arrange
        from src.scheduler.jobs.reminder import ReminderRecord, send_reminder

        record = ReminderRecord(
            id="rem-003", task_name="Task D", deadline="2026-04-26", owner="Dave", thread_id="t-4"
        )

        with patch("src.scheduler.jobs.reminder.settings") as mock_settings:
            mock_settings.notification_method = "webhook"
            mock_settings.notification_webhook_url = ""

            # Act
            result = await send_reminder(record)

        # Assert
        assert result is False

    @pytest.mark.asyncio
    async def test_runReminderJob_marksSentAfterSuccess(self):
        # Arrange
        from src.scheduler.jobs.reminder import ReminderRecord, run_reminder_job

        record = ReminderRecord(
            id="rem-001", task_name="Task", deadline="2026-04-26", owner="Eve", thread_id="t-5"
        )
        mock_repo = AsyncMock(spec=ReminderRepository)
        mock_repo.mark_sent = AsyncMock()

        with (
            patch("src.scheduler.jobs.reminder.fetch_pending_reminders", AsyncMock(return_value=[record])),
            patch("src.scheduler.jobs.reminder.send_reminder", AsyncMock(return_value=True)),
            patch("src.scheduler.jobs.reminder.AsyncSessionLocal") as mock_session_ctx,
            patch("src.scheduler.jobs.reminder.ReminderRepository", return_value=mock_repo),
        ):
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)
            mock_session_ctx.return_value = mock_session

            # Act
            summary = await run_reminder_job()

        # Assert
        assert summary["sent"] == 1
        assert summary["failed"] == 0
        mock_repo.mark_sent.assert_awaited_once_with("rem-001")

    @pytest.mark.asyncio
    async def test_runReminderJob_doesNotMarkSentOnFailure(self):
        # Arrange
        from src.scheduler.jobs.reminder import ReminderRecord, run_reminder_job

        record = ReminderRecord(
            id="rem-002", task_name="Task", deadline="2026-04-26", owner="Frank", thread_id="t-6"
        )
        mock_repo = AsyncMock(spec=ReminderRepository)
        mock_repo.mark_sent = AsyncMock()

        with (
            patch("src.scheduler.jobs.reminder.fetch_pending_reminders", AsyncMock(return_value=[record])),
            patch("src.scheduler.jobs.reminder.send_reminder", AsyncMock(return_value=False)),
            patch("src.scheduler.jobs.reminder.AsyncSessionLocal") as mock_session_ctx,
            patch("src.scheduler.jobs.reminder.ReminderRepository", return_value=mock_repo),
        ):
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)
            mock_session_ctx.return_value = mock_session

            # Act
            summary = await run_reminder_job()

        # Assert
        assert summary["sent"] == 0
        assert summary["failed"] == 1
        mock_repo.mark_sent.assert_not_awaited()


# ---------------------------------------------------------------------------
# TestQuotaResetJob
# ---------------------------------------------------------------------------

class TestQuotaResetJob:
    @pytest.mark.asyncio
    async def test_resetDaily_callsResetForAllActiveUsers(self):
        # Arrange
        from src.scheduler.jobs.quota_reset import reset_daily
        from src.persistence.repositories.quota_repo import QuotaRepository

        mock_repo = AsyncMock(spec=QuotaRepository)
        mock_repo.reset = AsyncMock()

        with (
            patch("src.scheduler.jobs.quota_reset._get_all_user_ids", AsyncMock(return_value=["u1", "u2"])),
            patch("src.scheduler.jobs.quota_reset.AsyncSessionLocal") as mock_session_ctx,
            patch("src.scheduler.jobs.quota_reset.QuotaRepository", return_value=mock_repo),
        ):
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)
            mock_session_ctx.return_value = mock_session

            # Act
            summary = await reset_daily()

        # Assert
        assert summary["period"] == "daily"
        assert summary["users_reset"] == 2
        assert mock_repo.reset.await_count == 2
        mock_repo.reset.assert_any_await("u1", "daily")
        mock_repo.reset.assert_any_await("u2", "daily")

    @pytest.mark.asyncio
    async def test_resetWeekly_callsResetForAllActiveUsers(self):
        # Arrange
        from src.scheduler.jobs.quota_reset import reset_weekly
        from src.persistence.repositories.quota_repo import QuotaRepository

        mock_repo = AsyncMock(spec=QuotaRepository)
        mock_repo.reset = AsyncMock()

        with (
            patch("src.scheduler.jobs.quota_reset._get_all_user_ids", AsyncMock(return_value=["u3"])),
            patch("src.scheduler.jobs.quota_reset.AsyncSessionLocal") as mock_session_ctx,
            patch("src.scheduler.jobs.quota_reset.QuotaRepository", return_value=mock_repo),
        ):
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)
            mock_session_ctx.return_value = mock_session

            # Act
            summary = await reset_weekly()

        # Assert
        assert summary["period"] == "weekly"
        assert summary["users_reset"] == 1
        mock_repo.reset.assert_awaited_once_with("u3", "weekly")

    @pytest.mark.asyncio
    async def test_resetDaily_noUsers_returnsZeroCount(self):
        # Arrange
        from src.scheduler.jobs.quota_reset import reset_daily
        from src.persistence.repositories.quota_repo import QuotaRepository

        mock_repo = AsyncMock(spec=QuotaRepository)

        with (
            patch("src.scheduler.jobs.quota_reset._get_all_user_ids", AsyncMock(return_value=[])),
            patch("src.scheduler.jobs.quota_reset.AsyncSessionLocal") as mock_session_ctx,
            patch("src.scheduler.jobs.quota_reset.QuotaRepository", return_value=mock_repo),
        ):
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)
            mock_session_ctx.return_value = mock_session

            # Act
            summary = await reset_daily()

        # Assert
        assert summary["users_reset"] == 0
        mock_repo.reset.assert_not_awaited()


# ---------------------------------------------------------------------------
# TestSchedulerSetup
# ---------------------------------------------------------------------------

class TestSchedulerSetup:
    def test_createScheduler_registersThreeJobs(self):
        # Act
        scheduler = create_scheduler()

        # Assert
        jobs = scheduler.get_jobs()
        assert len(jobs) == 3
        job_ids = {j.id for j in jobs}
        assert job_ids == {"reminder_job", "quota_reset_daily", "quota_reset_weekly"}

    def test_createScheduler_reminderJobIsIntervalTrigger(self):
        # Arrange
        from apscheduler.triggers.interval import IntervalTrigger

        # Act
        scheduler = create_scheduler()
        reminder_job = next(j for j in scheduler.get_jobs() if j.id == "reminder_job")

        # Assert
        assert isinstance(reminder_job.trigger, IntervalTrigger)

    def test_createScheduler_quotaResetJobIsCronTrigger(self):
        # Arrange
        from apscheduler.triggers.cron import CronTrigger

        # Act
        scheduler = create_scheduler()
        daily_job = next(j for j in scheduler.get_jobs() if j.id == "quota_reset_daily")
        weekly_job = next(j for j in scheduler.get_jobs() if j.id == "quota_reset_weekly")

        # Assert
        assert isinstance(daily_job.trigger, CronTrigger)
        assert isinstance(weekly_job.trigger, CronTrigger)
