import hashlib
from abc import ABC, abstractmethod
from typing import Optional


class LLMCache(ABC):
    @abstractmethod
    async def get(self, key: str) -> Optional[str]: ...

    @abstractmethod
    async def set(self, key: str, value: str, ttl: int = 3600) -> None: ...

    @abstractmethod
    async def delete(self, key: str) -> None: ...

    @abstractmethod
    async def clear(self) -> None: ...

    def make_key(self, prompt: str, model: str) -> str:
        digest = hashlib.sha256(f"{model}:{prompt}".encode()).hexdigest()
        return digest


class RedisLLMCache(LLMCache):
    def __init__(self, redis_url: str, prefix: str = "llm_cache"):
        self._redis_url = redis_url
        self._prefix = prefix
        self._client = None

    async def _get_client(self):
        if self._client is None:
            import redis.asyncio as aioredis
            self._client = aioredis.from_url(self._redis_url, decode_responses=True)
        return self._client

    def _full_key(self, key: str) -> str:
        return f"{self._prefix}:{key}"

    async def get(self, key: str) -> Optional[str]:
        client = await self._get_client()
        return await client.get(self._full_key(key))

    async def set(self, key: str, value: str, ttl: int = 3600) -> None:
        client = await self._get_client()
        await client.set(self._full_key(key), value, ex=ttl)

    async def delete(self, key: str) -> None:
        client = await self._get_client()
        await client.delete(self._full_key(key))

    async def clear(self) -> None:
        client = await self._get_client()
        pattern = f"{self._prefix}:*"
        keys = await client.keys(pattern)
        if keys:
            await client.delete(*keys)
