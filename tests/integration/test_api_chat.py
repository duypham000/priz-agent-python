"""Integration tests for Phase 8 REST API + SSE endpoints."""
import json
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.deps import get_checkpointer, get_current_user, get_db, get_llm_registry
from src.llm.mock import MockAdapter
from src.llm.registry import LLMRegistry
from src.main import app

# ── Constants ─────────────────────────────────────────────────────────────────

TEST_USER_ID = str(uuid.uuid4())
TEST_USER_EMAIL = "test@example.com"
TEST_THREAD_ID = str(uuid.uuid4())


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_user(user_id: str = TEST_USER_ID) -> MagicMock:
    user = MagicMock()
    user.id = user_id
    user.email = TEST_USER_EMAIL
    user.role = "user"
    user.is_active = True
    user.hashed_password = "hashed"
    user.created_at = datetime.now(timezone.utc)
    user.updated_at = datetime.now(timezone.utc)
    return user


def _make_thread(
    thread_id: str = TEST_THREAD_ID,
    user_id: str = TEST_USER_ID,
    status: str = "active",
) -> MagicMock:
    thread = MagicMock()
    thread.id = thread_id
    thread.user_id = user_id
    thread.title = "Test thread"
    thread.status = status
    thread.created_at = datetime.now(timezone.utc)
    thread.updated_at = datetime.now(timezone.utc)
    return thread


def _make_llm_registry() -> LLMRegistry:
    registry = LLMRegistry()
    mock = MockAdapter(responses=["This is a mock summary."])
    registry.register(mock, primary=True)
    registry.register_as("default", mock)
    return registry


def _parse_sse(text: str) -> list[dict]:
    """Parse SSE stream text → list of event dicts."""
    events = []
    for line in text.splitlines():
        if line.startswith("data: "):
            try:
                events.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                pass
    return events


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_user():
    return _make_user()


@pytest.fixture
def mock_thread():
    return _make_thread()


@pytest.fixture
def llm_registry():
    return _make_llm_registry()


@pytest.fixture
def mock_checkpointer():
    """Patch CheckpointerManager in main to bypass real PostgreSQL during lifespan."""
    instance = MagicMock()
    instance.__aenter__ = AsyncMock(return_value=instance)
    instance.__aexit__ = AsyncMock(return_value=None)
    instance.list_checkpoints = AsyncMock(return_value=[])
    with patch("src.main.CheckpointerManager", return_value=instance):
        yield instance


@pytest.fixture
def client_no_auth(mock_checkpointer):
    """TestClient with no auth overrides — for testing 401 responses."""
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
def client_with_mocks(mock_user, llm_registry, mock_checkpointer):
    """TestClient with all external dependencies overridden."""
    mock_db = AsyncMock()

    app.dependency_overrides[get_db] = lambda: mock_db
    app.dependency_overrides[get_current_user] = lambda: mock_user
    app.dependency_overrides[get_llm_registry] = lambda: llm_registry
    app.dependency_overrides[get_checkpointer] = lambda: mock_checkpointer

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c

    app.dependency_overrides.clear()


# ── Authentication ─────────────────────────────────────────────────────────────


class TestAuthentication:
    def test_chat_unauthenticated_returns_401(self, client_no_auth):
        resp = client_no_auth.post("/chat", json={"message": "hello"})
        assert resp.status_code == 401

    def test_sessions_unauthenticated_returns_401(self, client_no_auth):
        resp = client_no_auth.get("/sessions")
        assert resp.status_code == 401

    def test_quota_unauthenticated_returns_401(self, client_no_auth):
        resp = client_no_auth.get("/quota")
        assert resp.status_code == 401

    def test_models_unauthenticated_returns_401(self, client_no_auth):
        resp = client_no_auth.get("/models")
        assert resp.status_code == 401


# ── POST /chat ─────────────────────────────────────────────────────────────────


class TestChatEndpoint:
    def test_chat_creates_thread_and_streams_sse(self, client_with_mocks, mock_thread):
        mock_thread_repo = MagicMock()
        mock_thread_repo.get = AsyncMock(return_value=None)
        mock_thread_repo.create = AsyncMock(return_value=mock_thread)
        mock_thread_repo.update_status = AsyncMock()

        with patch("src.api.routers.chat.ThreadRepository", return_value=mock_thread_repo):
            resp = client_with_mocks.post("/chat", json={"message": "Summarize meeting notes."})

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        assert "x-thread-id" in resp.headers

        events = _parse_sse(resp.text)
        types = [e["type"] for e in events]
        assert "done" in types or "node_complete" in types

    def test_chat_with_existing_thread_id(self, client_with_mocks, mock_thread):
        mock_thread_repo = MagicMock()
        mock_thread_repo.get = AsyncMock(return_value=mock_thread)
        mock_thread_repo.update_status = AsyncMock()

        with patch("src.api.routers.chat.ThreadRepository", return_value=mock_thread_repo):
            resp = client_with_mocks.post(
                "/chat",
                json={"message": "Continue.", "thread_id": TEST_THREAD_ID},
            )

        assert resp.status_code == 200

    def test_chat_unknown_thread_returns_404(self, client_with_mocks):
        mock_thread_repo = MagicMock()
        mock_thread_repo.get = AsyncMock(return_value=None)

        with patch("src.api.routers.chat.ThreadRepository", return_value=mock_thread_repo):
            resp = client_with_mocks.post(
                "/chat",
                json={"message": "hello", "thread_id": str(uuid.uuid4())},
            )

        assert resp.status_code == 404

    def test_chat_unknown_agent_streams_error_event(self, client_with_mocks, mock_thread):
        mock_thread_repo = MagicMock()
        mock_thread_repo.get = AsyncMock(return_value=None)
        mock_thread_repo.create = AsyncMock(return_value=mock_thread)
        mock_thread_repo.update_status = AsyncMock()

        with patch("src.api.routers.chat.ThreadRepository", return_value=mock_thread_repo):
            resp = client_with_mocks.post(
                "/chat",
                json={"message": "hello", "agent_name": "nonexistent_agent"},
            )

        assert resp.status_code == 200
        events = _parse_sse(resp.text)
        assert any(e["type"] == "error" for e in events)


# ── POST /chat/{thread_id}/resume ──────────────────────────────────────────────


class TestResumeEndpoint:
    def test_resume_thread_awaiting_approval(self, client_with_mocks):
        waiting_thread = _make_thread(status="awaiting_approval")
        mock_thread_repo = MagicMock()
        mock_thread_repo.get = AsyncMock(return_value=waiting_thread)
        mock_thread_repo.update_status = AsyncMock()

        with patch("src.api.routers.chat.ThreadRepository", return_value=mock_thread_repo):
            resp = client_with_mocks.post(
                f"/chat/{TEST_THREAD_ID}/resume",
                json={"approved": True, "feedback": "Looks good."},
            )

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

    def test_resume_non_waiting_thread_returns_400(self, client_with_mocks, mock_thread):
        mock_thread_repo = MagicMock()
        mock_thread_repo.get = AsyncMock(return_value=mock_thread)

        with patch("src.api.routers.chat.ThreadRepository", return_value=mock_thread_repo):
            resp = client_with_mocks.post(
                f"/chat/{TEST_THREAD_ID}/resume",
                json={"approved": True},
            )

        assert resp.status_code == 400

    def test_resume_missing_thread_returns_404(self, client_with_mocks):
        mock_thread_repo = MagicMock()
        mock_thread_repo.get = AsyncMock(return_value=None)

        with patch("src.api.routers.chat.ThreadRepository", return_value=mock_thread_repo):
            resp = client_with_mocks.post(
                f"/chat/{str(uuid.uuid4())}/resume",
                json={"approved": False},
            )

        assert resp.status_code == 404


# ── GET /sessions ──────────────────────────────────────────────────────────────


class TestSessionsEndpoint:
    def test_get_sessions_returns_list(self, client_with_mocks, mock_thread):
        mock_thread_repo = MagicMock()
        mock_thread_repo.list_by_user = AsyncMock(return_value=[mock_thread])

        with patch("src.api.routers.sessions.ThreadRepository", return_value=mock_thread_repo):
            resp = client_with_mocks.get("/sessions")

        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert len(body["data"]["items"]) == 1
        assert body["data"]["items"][0]["id"] == TEST_THREAD_ID

    def test_get_session_by_id(self, client_with_mocks, mock_thread):
        mock_thread_repo = MagicMock()
        mock_thread_repo.get = AsyncMock(return_value=mock_thread)

        with patch("src.api.routers.sessions.ThreadRepository", return_value=mock_thread_repo):
            resp = client_with_mocks.get(f"/sessions/{TEST_THREAD_ID}")

        assert resp.status_code == 200
        body = resp.json()
        assert body["data"]["id"] == TEST_THREAD_ID

    def test_get_session_not_found_returns_404(self, client_with_mocks):
        mock_thread_repo = MagicMock()
        mock_thread_repo.get = AsyncMock(return_value=None)

        with patch("src.api.routers.sessions.ThreadRepository", return_value=mock_thread_repo):
            resp = client_with_mocks.get(f"/sessions/{str(uuid.uuid4())}")

        assert resp.status_code == 404

    def test_delete_session(self, client_with_mocks, mock_thread):
        mock_thread_repo = MagicMock()
        mock_thread_repo.get = AsyncMock(return_value=mock_thread)
        mock_thread_repo.delete = AsyncMock()

        with patch("src.api.routers.sessions.ThreadRepository", return_value=mock_thread_repo):
            resp = client_with_mocks.delete(f"/sessions/{TEST_THREAD_ID}")

        assert resp.status_code == 204
        mock_thread_repo.delete.assert_called_once_with(TEST_THREAD_ID)

    def test_get_checkpoints(self, client_with_mocks, mock_thread):
        mock_thread_repo = MagicMock()
        mock_thread_repo.get = AsyncMock(return_value=mock_thread)

        with patch("src.api.routers.sessions.ThreadRepository", return_value=mock_thread_repo):
            resp = client_with_mocks.get(f"/sessions/{TEST_THREAD_ID}/checkpoints")

        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert isinstance(body["data"], list)


# ── GET /quota ─────────────────────────────────────────────────────────────────


class TestQuotaEndpoint:
    def test_get_quota_returns_periods(self, client_with_mocks):
        mock_quota_repo = MagicMock()
        mock_quota_repo.get_usage = AsyncMock(return_value=[])

        with patch("src.api.routers.quota.QuotaRepository", return_value=mock_quota_repo):
            resp = client_with_mocks.get("/quota")

        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert "daily" in body["data"]
        assert "weekly" in body["data"]
        assert "monthly" in body["data"]

    def test_get_quota_aggregates_tokens_by_model(self, client_with_mocks):
        usage1 = MagicMock(tokens=100, model="mock-model")
        usage2 = MagicMock(tokens=50, model="mock-model")
        mock_quota_repo = MagicMock()
        mock_quota_repo.get_usage = AsyncMock(side_effect=[[usage1, usage2], [], []])

        with patch("src.api.routers.quota.QuotaRepository", return_value=mock_quota_repo):
            resp = client_with_mocks.get("/quota")

        body = resp.json()
        assert body["data"]["daily"]["total_tokens"] == 150
        assert body["data"]["daily"]["by_model"]["mock-model"] == 150


# ── GET /models ────────────────────────────────────────────────────────────────


class TestModelsEndpoint:
    def test_get_models_returns_list(self, client_with_mocks, llm_registry):
        resp = client_with_mocks.get("/models")

        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert len(body["data"]["models"]) > 0
        assert body["data"]["default"] is not None


# ── GET /health ────────────────────────────────────────────────────────────────


class TestHealthEndpoint:
    def test_health_check(self, mock_checkpointer):
        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
