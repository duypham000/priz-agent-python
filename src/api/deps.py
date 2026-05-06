import logging
from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from src.core.auth import TokenUser
from src.grpc.user_client import UserGrpcClient
from src.llm.registry import LLMRegistry
from src.persistence.checkpointer import CheckpointerManager
from src.persistence.database import get_session, AsyncSessionLocal
from src.persistence.models.user import User


async def get_db(session: AsyncSession = Depends(get_session)) -> AsyncSession:
    return session


async def get_llm_registry(request: Request) -> LLMRegistry:
    return request.app.state.llm_registry


async def get_checkpointer(request: Request) -> CheckpointerManager:
    checkpointer = request.app.state.checkpointer
    if checkpointer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Checkpointer service is currently unavailable",
        )
    return checkpointer


async def get_user_client(request: Request) -> UserGrpcClient:
    user_client = request.app.state.user_client
    if user_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="User gRPC service is currently unavailable",
        )
    return user_client


async def get_current_user(request: Request) -> TokenUser:
    # Phantom Token pattern: gateway has already validated the opaque token
    # and injected trusted user-info headers. No JWT validation needed here.
    user_id = request.headers.get("X-User-Id")
    if not user_id:
        logging.warning("Missing X-User-Id header — request did not pass through gateway auth")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )

    role = request.headers.get("X-User-Role", "USER")
    email = request.headers.get("X-User-Email", "")
    username = request.headers.get("X-User-Username", "")

    # JIT User Creation: ensure user exists in local DB for foreign key constraints
    async with AsyncSessionLocal() as session:
        async with session.begin():
            stmt = select(User).where(User.id == user_id)
            result = await session.execute(stmt)
            db_user = result.scalar_one_or_none()

            if not db_user:
                logging.info(f"Creating JIT user record for {user_id}")
                new_user = User(
                    id=user_id,
                    email=email,
                    role=role,
                    hashed_password=None,
                )
                session.add(new_user)
                await session.flush()

    return TokenUser(id=user_id, email=email, username=username, role=role)
