import logging
import time

import grpc
import grpc.aio
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from src.grpc.generated import user_service_pb2, user_service_pb2_grpc

logger = logging.getLogger(__name__)

_TRANSIENT_CODES = {grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED}
_CIRCUIT_COOLDOWN_S = 30.0
_GRPC_TIMEOUT_S = 3.0


def _is_transient(exc: BaseException) -> bool:
    return isinstance(exc, grpc.aio.AioRpcError) and exc.code() in _TRANSIENT_CODES


class UserGrpcClient:
    def __init__(self, address: str) -> None:
        self._address = address
        self._channel: grpc.aio.Channel | None = None
        self._circuit_open_until: float = 0.0

    def _is_circuit_open(self) -> bool:
        return time.monotonic() < self._circuit_open_until

    def _trip_circuit(self) -> None:
        self._circuit_open_until = time.monotonic() + _CIRCUIT_COOLDOWN_S
        logger.warning("gRPC circuit opened for %.0fs — base service may be down", _CIRCUIT_COOLDOWN_S)

    def _get_stub(self) -> user_service_pb2_grpc.UserServiceStub:
        if self._channel is None:
            self._channel = grpc.aio.insecure_channel(self._address)
        return user_service_pb2_grpc.UserServiceStub(self._channel)

    @retry(
        retry=retry_if_exception(_is_transient),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
        stop=stop_after_attempt(3),
        reraise=False,
    )
    async def get_user(self, user_id: str) -> user_service_pb2.UserInfo | None:
        if self._is_circuit_open():
            logger.debug("gRPC circuit open — skipping GetUser for user_id=%s", user_id)
            return None
        try:
            return await self._get_stub().GetUser(
                user_service_pb2.GetUserRequest(user_id=user_id),
                timeout=_GRPC_TIMEOUT_S,
            )
        except grpc.aio.AioRpcError as e:
            if e.code() in _TRANSIENT_CODES:
                self._trip_circuit()
            logger.warning("gRPC GetUser failed [%s] user_id=%s: %s", e.code(), user_id, e.details())
            return None

    @retry(
        retry=retry_if_exception(_is_transient),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
        stop=stop_after_attempt(3),
        reraise=False,
    )
    async def get_user_context(self, user_id: str) -> user_service_pb2.UserExecutionContext | None:
        if self._is_circuit_open():
            logger.debug("gRPC circuit open — skipping GetUserContext for user_id=%s", user_id)
            return None
        try:
            return await self._get_stub().GetUserContext(
                user_service_pb2.GetUserContextRequest(user_id=user_id),
                timeout=_GRPC_TIMEOUT_S,
            )
        except grpc.aio.AioRpcError as e:
            if e.code() in _TRANSIENT_CODES:
                self._trip_circuit()
            logger.warning("gRPC GetUserContext failed [%s] user_id=%s: %s", e.code(), user_id, e.details())
            return None

    async def close(self) -> None:
        if self._channel is not None:
            await self._channel.close()
            self._channel = None
