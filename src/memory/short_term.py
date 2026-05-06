import json
from typing import Optional

from langchain_core.messages import BaseMessage
from langchain_core.messages import messages_from_dict, message_to_dict


class ShortTermMemory:
    def __init__(self, redis_url: str, prefix: str = "short_term", ttl: int = 86400):
        self._redis_url = redis_url
        self._prefix = prefix
        self._ttl = ttl
        self._client = None

    async def _get_client(self):
        if self._client is None:
            import redis.asyncio as aioredis
            self._client = aioredis.from_url(self._redis_url, decode_responses=True)
        return self._client

    def _key(self, thread_id: str) -> str:
        return f"{self._prefix}:{thread_id}"

    async def add_message(self, thread_id: str, msg: BaseMessage) -> None:
        client = await self._get_client()
        key = self._key(thread_id)
        serialized = json.dumps(message_to_dict(msg))
        await client.rpush(key, serialized)
        await client.expire(key, self._ttl)

    async def get_recent(self, thread_id: str, n: int = 10) -> list[BaseMessage]:
        client = await self._get_client()
        key = self._key(thread_id)
        items = await client.lrange(key, -n, -1)
        dicts = [json.loads(item) for item in items]
        return messages_from_dict(dicts)

    async def clear(self, thread_id: str) -> None:
        client = await self._get_client()
        await client.delete(self._key(thread_id))
