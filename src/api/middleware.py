import logging
import time

import redis.asyncio as aioredis
from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.core.exceptions import AgentError, QuotaExceededError
from src.core.response import ApiResponse

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "%s %s → %d (%.1fms)",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Redis sliding-window rate limiter applied to /chat endpoints."""

    def __init__(self, app, redis_url: str, limit: int, window_seconds: int = 60):
        super().__init__(app)
        self._redis_url = redis_url
        self._limit = limit
        self._window = window_seconds

    async def dispatch(self, request: Request, call_next):
        if not request.url.path.startswith("/chat"):
            return await call_next(request)

        user_id = self._extract_user_id(request)
        if user_id is None:
            return await call_next(request)

        allowed = await self._check_rate(user_id)
        if not allowed:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content=ApiResponse.error("Rate limit exceeded. Try again in a moment.").model_dump(),
            )
        return await call_next(request)

    def _extract_user_id(self, request: Request) -> str | None:
        # Gateway injects X-User-Id after Phantom Token exchange
        return request.headers.get("X-User-Id")

    async def _check_rate(self, user_id: str) -> bool:
        try:
            now = time.time()
            window_start = now - self._window
            key = f"ratelimit:{user_id}"
            async with aioredis.from_url(self._redis_url) as r:
                pipe = r.pipeline()
                pipe.zremrangebyscore(key, 0, window_start)
                pipe.zadd(key, {str(now): now})
                pipe.zcard(key)
                pipe.expire(key, self._window + 5)
                results = await pipe.execute()
            count = results[2]
            return count <= self._limit
        except Exception as e:
            logger.warning(f"Rate limiter failed (ignoring): {e}")
            return True


async def agent_error_handler(request: Request, exc: AgentError) -> JSONResponse:
    if isinstance(exc, QuotaExceededError):
        http_status = status.HTTP_429_TOO_MANY_REQUESTS
    else:
        http_status = status.HTTP_500_INTERNAL_SERVER_ERROR
    return JSONResponse(
        status_code=http_status,
        content=ApiResponse.error(str(exc)).model_dump(),
    )


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ApiResponse.error(str(exc)).model_dump(),
    )
