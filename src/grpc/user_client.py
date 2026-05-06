import logging

import grpc
import grpc.aio

from src.grpc.generated import user_service_pb2, user_service_pb2_grpc

logger = logging.getLogger(__name__)


class UserGrpcClient:
    def __init__(self, address: str) -> None:
        self._address = address
        self._channel: grpc.aio.Channel | None = None

    def _get_stub(self) -> user_service_pb2_grpc.UserServiceStub:
        if self._channel is None:
            self._channel = grpc.aio.insecure_channel(self._address)
        return user_service_pb2_grpc.UserServiceStub(self._channel)

    async def get_user(self, user_id: str) -> user_service_pb2.UserInfo:
        try:
            stub = self._get_stub()
            return await stub.GetUser(user_service_pb2.GetUserRequest(user_id=user_id))
        except grpc.aio.AioRpcError as e:
            logger.error("gRPC GetUser failed for user_id=%s: %s", user_id, e.details())
            raise RuntimeError(f"Failed to fetch user from base: {e.details()}") from e

    async def get_user_context(self, user_id: str) -> user_service_pb2.UserExecutionContext:
        try:
            stub = self._get_stub()
            return await stub.GetUserContext(user_service_pb2.GetUserContextRequest(user_id=user_id))
        except grpc.aio.AioRpcError as e:
            logger.error("gRPC GetUserContext failed for user_id=%s: %s", user_id, e.details())
            raise RuntimeError(f"Failed to fetch user context from base: {e.details()}") from e

    async def close(self) -> None:
        if self._channel is not None:
            await self._channel.close()
            self._channel = None
