import json
from typing import Any, Optional


class EntityMemory:
    def __init__(self, redis_url: str, prefix: str = "entity", ttl: int = 604800):
        self._redis_url = redis_url
        self._prefix = prefix
        self._ttl = ttl
        self._client = None

    async def _get_client(self):
        if self._client is None:
            import redis.asyncio as aioredis
            self._client = aioredis.from_url(self._redis_url, decode_responses=True)
        return self._client

    def _key(self, user_id: str) -> str:
        return f"{self._prefix}:{user_id}"

    async def set(self, user_id: str, key: str, value: Any) -> None:
        client = await self._get_client()
        hash_key = self._key(user_id)
        await client.hset(hash_key, key, json.dumps(value))
        await client.expire(hash_key, self._ttl)

    async def get(self, user_id: str, key: str) -> Optional[Any]:
        client = await self._get_client()
        raw = await client.hget(self._key(user_id), key)
        if raw is None:
            return None
        return json.loads(raw)

    async def get_all(self, user_id: str) -> dict[str, Any]:
        client = await self._get_client()
        raw = await client.hgetall(self._key(user_id))
        return {k: json.loads(v) for k, v in raw.items()}

    async def delete(self, user_id: str, key: str) -> None:
        client = await self._get_client()
        await client.hdel(self._key(user_id), key)
