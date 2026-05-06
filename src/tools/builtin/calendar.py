from __future__ import annotations

import base64
import json
import logging
from typing import Protocol

import httpx

from src.core.exceptions import ToolError
from src.settings import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------


class CalendarBackend(Protocol):
    async def create_task(self, title: str, deadline: str, owner: str) -> dict: ...
    async def list_tasks(self) -> list[dict]: ...


# ---------------------------------------------------------------------------
# Log backend (default stub)
# ---------------------------------------------------------------------------


class LogBackend:
    async def create_task(self, title: str, deadline: str, owner: str) -> dict:
        logger.info("calendar[log] create_task: title=%r deadline=%s owner=%s", title, deadline, owner)
        return {"status": "logged", "title": title, "deadline": deadline, "owner": owner}

    async def list_tasks(self) -> list[dict]:
        logger.info("calendar[log] list_tasks called")
        return []


# ---------------------------------------------------------------------------
# Notion backend
# ---------------------------------------------------------------------------


class NotionBackend:
    _BASE_URL = "https://api.notion.com/v1"
    _VERSION = "2022-06-28"

    def __init__(self) -> None:
        if not settings.notion_api_key:
            raise ToolError("Notion API key not configured", code="NOTION_NOT_CONFIGURED")
        if not settings.notion_database_id:
            raise ToolError("Notion database ID not configured", code="NOTION_NOT_CONFIGURED")
        self._headers = {
            "Authorization": f"Bearer {settings.notion_api_key}",
            "Notion-Version": self._VERSION,
            "Content-Type": "application/json",
        }

    async def create_task(self, title: str, deadline: str, owner: str) -> dict:
        payload: dict = {
            "parent": {"database_id": settings.notion_database_id},
            "properties": {
                "Name": {"title": [{"text": {"content": title}}]},
                "Owner": {"rich_text": [{"text": {"content": owner}}]},
            },
        }
        if deadline and deadline != "TBD":
            payload["properties"]["Deadline"] = {"date": {"start": deadline}}

        async with httpx.AsyncClient() as client:
            try:
                resp = await client.post(
                    f"{self._BASE_URL}/pages",
                    headers=self._headers,
                    json=payload,
                    timeout=10.0,
                )
                resp.raise_for_status()
                data = resp.json()
                return {"status": "synced", "id": data.get("id"), "title": title}
            except httpx.HTTPStatusError as exc:
                raise ToolError(
                    f"Notion API error {exc.response.status_code}: {exc.response.text}",
                    code="NOTION_API_ERROR",
                ) from exc

    async def list_tasks(self) -> list[dict]:
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.post(
                    f"{self._BASE_URL}/databases/{settings.notion_database_id}/query",
                    headers=self._headers,
                    json={},
                    timeout=10.0,
                )
                resp.raise_for_status()
                results = resp.json().get("results", [])
                return [
                    {"id": r["id"], "title": _extract_notion_title(r)}
                    for r in results
                ]
            except httpx.HTTPStatusError as exc:
                raise ToolError(
                    f"Notion API error {exc.response.status_code}",
                    code="NOTION_API_ERROR",
                ) from exc


def _extract_notion_title(page: dict) -> str:
    try:
        return page["properties"]["Name"]["title"][0]["text"]["content"]
    except (KeyError, IndexError):
        return ""


# ---------------------------------------------------------------------------
# Jira backend
# ---------------------------------------------------------------------------


class JiraBackend:
    def __init__(self) -> None:
        if not settings.jira_base_url:
            raise ToolError("Jira base URL not configured", code="JIRA_NOT_CONFIGURED")
        if not settings.jira_api_token:
            raise ToolError("Jira API token not configured", code="JIRA_NOT_CONFIGURED")
        credentials = base64.b64encode(
            f"{settings.jira_email}:{settings.jira_api_token}".encode()
        ).decode()
        self._headers = {
            "Authorization": f"Basic {credentials}",
            "Content-Type": "application/json",
        }
        self._base_url = settings.jira_base_url.rstrip("/")

    async def create_task(self, title: str, deadline: str, owner: str) -> dict:
        payload: dict = {
            "fields": {
                "project": {"key": settings.jira_project_key},
                "summary": title,
                "issuetype": {"name": "Task"},
            }
        }
        if deadline and deadline != "TBD":
            payload["fields"]["duedate"] = deadline
        if owner and owner != "Unassigned":
            payload["fields"]["assignee"] = {"name": owner}

        async with httpx.AsyncClient() as client:
            try:
                resp = await client.post(
                    f"{self._base_url}/rest/api/2/issue",
                    headers=self._headers,
                    json=payload,
                    timeout=10.0,
                )
                resp.raise_for_status()
                data = resp.json()
                return {"status": "synced", "id": data.get("id"), "key": data.get("key"), "title": title}
            except httpx.HTTPStatusError as exc:
                raise ToolError(
                    f"Jira API error {exc.response.status_code}: {exc.response.text}",
                    code="JIRA_API_ERROR",
                ) from exc

    async def list_tasks(self) -> list[dict]:
        jql = f"project={settings.jira_project_key} ORDER BY created DESC"
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(
                    f"{self._base_url}/rest/api/2/search",
                    headers=self._headers,
                    params={"jql": jql, "maxResults": 50},
                    timeout=10.0,
                )
                resp.raise_for_status()
                issues = resp.json().get("issues", [])
                return [
                    {"id": i["id"], "key": i["key"], "title": i["fields"]["summary"]}
                    for i in issues
                ]
            except httpx.HTTPStatusError as exc:
                raise ToolError(
                    f"Jira API error {exc.response.status_code}",
                    code="JIRA_API_ERROR",
                ) from exc


# ---------------------------------------------------------------------------
# Google Calendar backend
# ---------------------------------------------------------------------------


class GoogleCalendarBackend:
    def __init__(self) -> None:
        if not settings.google_service_account_json:
            raise ToolError(
                "Google service account JSON not configured",
                code="GOOGLE_NOT_CONFIGURED",
            )
        try:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build

            sa_info = json.loads(settings.google_service_account_json)
            creds = service_account.Credentials.from_service_account_info(
                sa_info,
                scopes=["https://www.googleapis.com/auth/calendar"],
            )
            self._service = build("calendar", "v3", credentials=creds, cache_discovery=False)
        except ImportError as exc:
            raise ToolError(
                "google-api-python-client not installed. Run: uv add google-api-python-client google-auth",
                code="GOOGLE_NOT_INSTALLED",
            ) from exc

    async def create_task(self, title: str, deadline: str, owner: str) -> dict:
        start_date = deadline if (deadline and deadline != "TBD") else None
        event: dict = {
            "summary": title,
            "description": f"Owner: {owner}",
        }
        if start_date:
            event["start"] = {"date": start_date}
            event["end"] = {"date": start_date}
        else:
            from datetime import date
            today = str(date.today())
            event["start"] = {"date": today}
            event["end"] = {"date": today}

        try:
            result = (
                self._service.events()
                .insert(calendarId=settings.google_calendar_id, body=event)
                .execute()
            )
            return {"status": "synced", "id": result.get("id"), "title": title}
        except Exception as exc:
            raise ToolError(f"Google Calendar API error: {exc}", code="GOOGLE_API_ERROR") from exc

    async def list_tasks(self) -> list[dict]:
        try:
            result = (
                self._service.events()
                .list(calendarId=settings.google_calendar_id, maxResults=50)
                .execute()
            )
            events = result.get("items", [])
            return [{"id": e["id"], "title": e.get("summary", "")} for e in events]
        except Exception as exc:
            raise ToolError(f"Google Calendar API error: {exc}", code="GOOGLE_API_ERROR") from exc


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------

_BACKENDS: dict[str, type] = {
    "log": LogBackend,
    "notion": NotionBackend,
    "jira": JiraBackend,
    "google": GoogleCalendarBackend,
}


def get_backend() -> CalendarBackend:
    name = settings.calendar_backend.lower()
    backend_cls = _BACKENDS.get(name)
    if backend_cls is None:
        logger.warning("Unknown calendar_backend=%r, falling back to log", name)
        return LogBackend()
    return backend_cls()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def create_task(title: str, deadline: str, owner: str) -> dict:
    return await get_backend().create_task(title, deadline, owner)


async def list_tasks() -> list[dict]:
    return await get_backend().list_tasks()
