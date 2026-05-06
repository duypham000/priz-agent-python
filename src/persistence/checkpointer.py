from typing import Any

from langgraph.checkpoint.base import CheckpointTuple
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver


class CheckpointerManager:
    """Async context manager wrapping AsyncPostgresSaver for LangGraph state persistence."""

    def __init__(self, postgres_url: str) -> None:
        # AsyncPostgresSaver requires psycopg3 URL format (no +asyncpg driver prefix)
        self._psycopg_url = postgres_url.replace("postgresql+asyncpg://", "postgresql://")
        self._saver: AsyncPostgresSaver | None = None
        self._conn_ctx: Any = None

    async def __aenter__(self) -> "CheckpointerManager":
        # from_conn_string is an async context manager, not a coroutine
        self._conn_ctx = AsyncPostgresSaver.from_conn_string(self._psycopg_url)
        self._saver = await self._conn_ctx.__aenter__()
        await self._saver.setup()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._conn_ctx is not None:
            await self._conn_ctx.__aexit__(exc_type, exc_val, exc_tb)
            self._conn_ctx = None
            self._saver = None

    @property
    def saver(self) -> AsyncPostgresSaver:
        if self._saver is None:
            raise RuntimeError("CheckpointerManager must be used as an async context manager")
        return self._saver

    async def list_checkpoints(self, thread_id: str) -> list[CheckpointTuple]:
        config = {"configurable": {"thread_id": thread_id}}
        return [c async for c in self.saver.alist(config)]

    async def get_checkpoint(self, thread_id: str, checkpoint_id: str) -> CheckpointTuple | None:
        config = {"configurable": {"thread_id": thread_id, "checkpoint_id": checkpoint_id}}
        return await self.saver.aget_tuple(config)
