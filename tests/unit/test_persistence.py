from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.persistence.models.quota import TokenUsage
from src.persistence.models.session import Thread
from src.persistence.repositories.quota_repo import QuotaRepository
from src.persistence.repositories.session_repo import ThreadRepository

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TEST_USER_ID = "user-001"
TEST_THREAD_ID = "thread-abc"
_NOW = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_thread(**kwargs) -> MagicMock:
    defaults = dict(id=TEST_THREAD_ID, user_id=TEST_USER_ID, title="Test", status="active", created_at=_NOW, updated_at=_NOW)
    defaults.update(kwargs)
    t = MagicMock(spec=Thread)
    for k, v in defaults.items():
        setattr(t, k, v)
    return t


def _make_usage(**kwargs) -> MagicMock:
    defaults = dict(id="usage-001", user_id=TEST_USER_ID, model="gemini-pro", tokens=500, period="daily", recorded_at=_NOW)
    defaults.update(kwargs)
    u = MagicMock(spec=TokenUsage)
    for k, v in defaults.items():
        setattr(u, k, v)
    return u


# ---------------------------------------------------------------------------
# TestThreadRepository
# ---------------------------------------------------------------------------
class TestThreadRepository:
    def _make_repo(self) -> tuple[ThreadRepository, AsyncMock]:
        session = AsyncMock()
        return ThreadRepository(session), session

    @pytest.mark.asyncio
    async def test_create_addsAndReturnsThread(self):
        # Arrange
        repo, session = self._make_repo()
        thread = _make_thread()
        session.refresh = AsyncMock(side_effect=lambda t: None)
        session.add = MagicMock()
        session.commit = AsyncMock()

        with patch("src.persistence.repositories.session_repo.Thread", return_value=thread):
            # Act
            result = await repo.create(user_id=TEST_USER_ID, title="Test")

        # Assert
        session.add.assert_called_once()
        session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_existingThread_returnsThread(self):
        # Arrange
        repo, session = self._make_repo()
        thread = _make_thread()
        scalar_result = MagicMock()
        scalar_result.scalar_one_or_none.return_value = thread
        session.execute = AsyncMock(return_value=scalar_result)

        # Act
        result = await repo.get(TEST_THREAD_ID)

        # Assert
        assert result is thread

    @pytest.mark.asyncio
    async def test_get_nonExistingThread_returnsNone(self):
        # Arrange
        repo, session = self._make_repo()
        scalar_result = MagicMock()
        scalar_result.scalar_one_or_none.return_value = None
        session.execute = AsyncMock(return_value=scalar_result)

        # Act
        result = await repo.get("nonexistent-id")

        # Assert
        assert result is None

    @pytest.mark.asyncio
    async def test_listByUser_returnsOrderedList(self):
        # Arrange
        repo, session = self._make_repo()
        threads = [_make_thread(id=f"t{i}") for i in range(3)]
        scalars_result = MagicMock()
        scalars_result.scalars.return_value.all.return_value = threads
        session.execute = AsyncMock(return_value=scalars_result)

        # Act
        result = await repo.list_by_user(TEST_USER_ID, limit=10, offset=0)

        # Assert
        assert result == threads

    @pytest.mark.asyncio
    async def test_updateStatus_existingThread_updatesAndReturns(self):
        # Arrange
        repo, session = self._make_repo()
        thread = _make_thread(status="active")
        scalar_result = MagicMock()
        scalar_result.scalar_one_or_none.return_value = thread
        session.execute = AsyncMock(return_value=scalar_result)
        session.commit = AsyncMock()
        session.refresh = AsyncMock()

        # Act
        result = await repo.update_status(TEST_THREAD_ID, "completed")

        # Assert
        assert thread.status == "completed"
        session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_updateStatus_nonExistingThread_returnsNone(self):
        # Arrange
        repo, session = self._make_repo()
        scalar_result = MagicMock()
        scalar_result.scalar_one_or_none.return_value = None
        session.execute = AsyncMock(return_value=scalar_result)

        # Act
        result = await repo.update_status("ghost-id", "paused")

        # Assert
        assert result is None
        session.commit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_delete_existingThread_deletesAndCommits(self):
        # Arrange
        repo, session = self._make_repo()
        thread = _make_thread()
        scalar_result = MagicMock()
        scalar_result.scalar_one_or_none.return_value = thread
        session.execute = AsyncMock(return_value=scalar_result)
        session.delete = AsyncMock()
        session.commit = AsyncMock()

        # Act
        await repo.delete(TEST_THREAD_ID)

        # Assert
        session.delete.assert_awaited_once_with(thread)
        session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_delete_nonExistingThread_doesNothing(self):
        # Arrange
        repo, session = self._make_repo()
        scalar_result = MagicMock()
        scalar_result.scalar_one_or_none.return_value = None
        session.execute = AsyncMock(return_value=scalar_result)
        session.delete = AsyncMock()

        # Act
        await repo.delete("ghost-id")

        # Assert
        session.delete.assert_not_awaited()


# ---------------------------------------------------------------------------
# TestQuotaRepository
# ---------------------------------------------------------------------------
class TestQuotaRepository:
    def _make_repo(self) -> tuple[QuotaRepository, AsyncMock]:
        session = AsyncMock()
        return QuotaRepository(session), session

    @pytest.mark.asyncio
    async def test_recordUsage_createsAndReturnsUsage(self):
        # Arrange
        repo, session = self._make_repo()
        usage = _make_usage()
        session.add = MagicMock()
        session.commit = AsyncMock()
        session.refresh = AsyncMock()

        with patch("src.persistence.repositories.quota_repo.TokenUsage", return_value=usage):
            # Act
            result = await repo.record_usage(TEST_USER_ID, "gemini-pro", 500, "daily")

        # Assert
        session.add.assert_called_once()
        session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_getUsage_returnsListForPeriod(self):
        # Arrange
        repo, session = self._make_repo()
        usages = [_make_usage(id=f"u{i}") for i in range(2)]
        scalars_result = MagicMock()
        scalars_result.scalars.return_value.all.return_value = usages
        session.execute = AsyncMock(return_value=scalars_result)

        # Act
        result = await repo.get_usage(TEST_USER_ID, "daily")

        # Assert
        assert result == usages

    @pytest.mark.asyncio
    async def test_reset_deletesRecordsForPeriod(self):
        # Arrange
        repo, session = self._make_repo()
        session.execute = AsyncMock()
        session.commit = AsyncMock()

        # Act
        await repo.reset(TEST_USER_ID, "daily")

        # Assert
        session.execute.assert_awaited_once()
        session.commit.assert_awaited_once()


# ---------------------------------------------------------------------------
# TestCheckpointerManager
# ---------------------------------------------------------------------------
class TestCheckpointerManager:
    @pytest.mark.asyncio
    async def test_urlConversion_stripsAsyncpgPrefix(self):
        # Arrange
        from src.persistence.checkpointer import CheckpointerManager

        # Act
        mgr = CheckpointerManager("postgresql+asyncpg://user:pass@localhost/db")

        # Assert
        assert mgr._psycopg_url == "postgresql://user:pass@localhost/db"

    @pytest.mark.asyncio
    async def test_saverProperty_beforeEnter_raisesRuntimeError(self):
        # Arrange
        from src.persistence.checkpointer import CheckpointerManager

        mgr = CheckpointerManager("postgresql+asyncpg://user:pass@localhost/db")

        # Act & Assert
        with pytest.raises(RuntimeError, match="async context manager"):
            _ = mgr.saver

    @pytest.mark.asyncio
    async def test_listCheckpoints_delegatesToSaver(self):
        # Arrange
        from src.persistence.checkpointer import CheckpointerManager

        mock_saver = MagicMock()
        checkpoint = MagicMock()

        async def _alist(config):
            yield checkpoint

        mock_saver.alist = _alist

        mgr = CheckpointerManager("postgresql+asyncpg://user:pass@localhost/db")
        mgr._saver = mock_saver

        # Act
        result = await mgr.list_checkpoints("thread-001")

        # Assert
        assert result == [checkpoint]

    @pytest.mark.asyncio
    async def test_getCheckpoint_delegatesToSaver(self):
        # Arrange
        from src.persistence.checkpointer import CheckpointerManager

        mock_saver = MagicMock()
        checkpoint_tuple = MagicMock()
        mock_saver.aget_tuple = AsyncMock(return_value=checkpoint_tuple)

        mgr = CheckpointerManager("postgresql+asyncpg://user:pass@localhost/db")
        mgr._saver = mock_saver

        # Act
        result = await mgr.get_checkpoint("thread-001", "cp-abc")

        # Assert
        assert result is checkpoint_tuple
        mock_saver.aget_tuple.assert_awaited_once_with(
            {"configurable": {"thread_id": "thread-001", "checkpoint_id": "cp-abc"}}
        )
