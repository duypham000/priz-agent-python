from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.exceptions import QuotaExceededError
from src.persistence.models.quota import TokenUsage
from src.persistence.repositories.quota_repo import QuotaRepository

# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------

TEST_USER_ID = "user-001"
TEST_MODEL = "mock"
_NOW = datetime.now(timezone.utc)

_DAILY_LIMIT = 1000
_WEEKLY_LIMIT = 5000
_MONTHLY_LIMIT = 20000


def _make_usage(tokens: int, period: str = "daily", model: str = TEST_MODEL) -> MagicMock:
    u = MagicMock(spec=TokenUsage)
    u.id = "usage-001"
    u.user_id = TEST_USER_ID
    u.model = model
    u.tokens = tokens
    u.period = period
    u.recorded_at = _NOW
    return u


def _make_repo() -> tuple[QuotaRepository, AsyncMock]:
    session = AsyncMock()
    return QuotaRepository(session), session


# ---------------------------------------------------------------------------
# QuotaRepository tests
# ---------------------------------------------------------------------------

class TestQuotaRepository:

    @pytest.mark.asyncio
    async def test_record_usage_validData_persistsRecord(self):
        # Arrange
        repo, session = _make_repo()
        saved = _make_usage(tokens=100)
        session.refresh = AsyncMock()

        async def fake_refresh(obj):
            obj.id = saved.id

        session.refresh.side_effect = fake_refresh

        # Act
        result = await repo.record_usage(TEST_USER_ID, TEST_MODEL, 100, "daily")

        # Assert
        session.add.assert_called_once()
        session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_total_usage_multipleRecords_sumsCorrectly(self):
        # Arrange
        repo, _ = _make_repo()
        usages = [_make_usage(300), _make_usage(200), _make_usage(100)]
        repo.get_usage = AsyncMock(return_value=usages)

        # Act
        total = await repo.get_total_usage(TEST_USER_ID, "daily")

        # Assert
        assert total == 600

    @pytest.mark.asyncio
    async def test_get_total_usage_noRecords_returnsZero(self):
        # Arrange
        repo, _ = _make_repo()
        repo.get_usage = AsyncMock(return_value=[])

        # Act
        total = await repo.get_total_usage(TEST_USER_ID, "daily")

        # Assert
        assert total == 0

    @pytest.mark.asyncio
    async def test_check_limit_underAllLimits_doesNotRaise(self):
        # Arrange
        repo, _ = _make_repo()
        repo.get_total_usage = AsyncMock(return_value=100)

        fake_settings = MagicMock()
        fake_settings.quota_daily_token_limit = _DAILY_LIMIT
        fake_settings.quota_weekly_token_limit = _WEEKLY_LIMIT
        fake_settings.quota_monthly_token_limit = _MONTHLY_LIMIT

        # Act & Assert — should not raise
        with patch("src.persistence.repositories.quota_repo.settings", fake_settings):
            await repo.check_limit(TEST_USER_ID, TEST_MODEL, additional_tokens=50)

    @pytest.mark.asyncio
    async def test_check_limit_overDailyLimit_raisesQuotaExceededError(self):
        # Arrange
        repo, _ = _make_repo()

        async def fake_total(user_id: str, period: str) -> int:
            return {"daily": 950, "weekly": 950, "monthly": 950}[period]

        repo.get_total_usage = fake_total

        fake_settings = MagicMock()
        fake_settings.quota_daily_token_limit = _DAILY_LIMIT       # 1000
        fake_settings.quota_weekly_token_limit = _WEEKLY_LIMIT
        fake_settings.quota_monthly_token_limit = _MONTHLY_LIMIT

        # Act & Assert
        with patch("src.persistence.repositories.quota_repo.settings", fake_settings):
            with pytest.raises(QuotaExceededError) as exc_info:
                await repo.check_limit(TEST_USER_ID, TEST_MODEL, additional_tokens=100)

        err = exc_info.value
        assert err.model == TEST_MODEL
        assert err.limit == _DAILY_LIMIT
        assert err.usage == 1050

    @pytest.mark.asyncio
    async def test_check_limit_overWeeklyLimit_raisesQuotaExceededError(self):
        # Arrange
        repo, _ = _make_repo()

        async def fake_total(user_id: str, period: str) -> int:
            return {"daily": 100, "weekly": 4950, "monthly": 4950}[period]

        repo.get_total_usage = fake_total

        fake_settings = MagicMock()
        fake_settings.quota_daily_token_limit = _DAILY_LIMIT
        fake_settings.quota_weekly_token_limit = _WEEKLY_LIMIT       # 5000
        fake_settings.quota_monthly_token_limit = _MONTHLY_LIMIT

        # Act & Assert
        with patch("src.persistence.repositories.quota_repo.settings", fake_settings):
            with pytest.raises(QuotaExceededError) as exc_info:
                await repo.check_limit(TEST_USER_ID, TEST_MODEL, additional_tokens=100)

        err = exc_info.value
        assert err.limit == _WEEKLY_LIMIT

    @pytest.mark.asyncio
    async def test_check_limit_overMonthlyLimit_raisesQuotaExceededError(self):
        # Arrange
        repo, _ = _make_repo()

        async def fake_total(user_id: str, period: str) -> int:
            return {"daily": 100, "weekly": 100, "monthly": 19950}[period]

        repo.get_total_usage = fake_total

        fake_settings = MagicMock()
        fake_settings.quota_daily_token_limit = _DAILY_LIMIT
        fake_settings.quota_weekly_token_limit = _WEEKLY_LIMIT
        fake_settings.quota_monthly_token_limit = _MONTHLY_LIMIT     # 20000

        # Act & Assert
        with patch("src.persistence.repositories.quota_repo.settings", fake_settings):
            with pytest.raises(QuotaExceededError) as exc_info:
                await repo.check_limit(TEST_USER_ID, TEST_MODEL, additional_tokens=100)

        err = exc_info.value
        assert err.limit == _MONTHLY_LIMIT

    @pytest.mark.asyncio
    async def test_check_limit_exactlyAtLimit_doesNotRaise(self):
        # Arrange
        repo, _ = _make_repo()
        repo.get_total_usage = AsyncMock(return_value=_DAILY_LIMIT)

        fake_settings = MagicMock()
        fake_settings.quota_daily_token_limit = _DAILY_LIMIT
        fake_settings.quota_weekly_token_limit = _WEEKLY_LIMIT
        fake_settings.quota_monthly_token_limit = _MONTHLY_LIMIT

        # Act & Assert — exactly at limit (not over) should NOT raise
        with patch("src.persistence.repositories.quota_repo.settings", fake_settings):
            await repo.check_limit(TEST_USER_ID, TEST_MODEL, additional_tokens=0)


# ---------------------------------------------------------------------------
# RateLimitMiddleware tests
# ---------------------------------------------------------------------------

class TestRateLimitMiddleware:

    def _make_middleware(self, limit: int = 10):
        from src.api.middleware import RateLimitMiddleware
        app = MagicMock()
        middleware = RateLimitMiddleware.__new__(RateLimitMiddleware)
        middleware._redis_url = "redis://localhost:6389/0"
        middleware._limit = limit
        middleware._window = 60
        return middleware

    def test_extract_user_id_validToken_returnsSubject(self):
        # Arrange
        from jose import jwt as _jwt
        middleware = self._make_middleware()
        secret = "test-secret-at-least-32-chars-long!!"
        token = _jwt.encode({"sub": "user-123"}, secret, algorithm="HS256")

        request = MagicMock()
        request.headers.get.return_value = f"Bearer {token}"

        # Act — patch module-level settings in middleware
        with patch("src.api.middleware.settings") as fake_settings:
            fake_settings.jwt_secret = secret
            result = middleware._extract_user_id(request)

        # Assert
        assert result == "user-123"

    def test_extract_user_id_missingBearer_returnsNone(self):
        # Arrange
        middleware = self._make_middleware()
        request = MagicMock()
        request.headers.get.return_value = ""

        # Act
        result = middleware._extract_user_id(request)

        # Assert
        assert result is None

    def test_extract_user_id_invalidToken_returnsNone(self):
        # Arrange
        middleware = self._make_middleware()
        request = MagicMock()
        request.headers.get.return_value = "Bearer not-a-valid-token"

        # Act
        with patch("src.api.middleware.settings") as fake_settings:
            fake_settings.jwt_secret = "some-secret"
            result = middleware._extract_user_id(request)

        # Assert
        assert result is None

    @pytest.mark.asyncio
    async def test_dispatch_nonChatPath_skipsRateLimit(self):
        # Arrange
        from src.api.middleware import RateLimitMiddleware

        app = MagicMock()
        middleware = RateLimitMiddleware.__new__(RateLimitMiddleware)
        middleware._redis_url = "redis://localhost:6389/0"
        middleware._limit = 5
        middleware._window = 60

        expected_response = MagicMock()

        async def call_next(req):
            return expected_response

        request = MagicMock()
        request.url.path = "/quota"

        # Act
        result = await middleware.dispatch(request, call_next)

        # Assert — should pass through without rate limiting
        assert result == expected_response
