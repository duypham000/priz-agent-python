import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from src.api.middleware import LoggingMiddleware, RateLimitMiddleware, agent_error_handler, validation_error_handler
from src.grpc.user_client import UserGrpcClient
from src.api.routers import health
from src.api.routers import chat, models, quota, sessions, agents
from src.api.routers import metrics as metrics_router
from src.core.exceptions import AgentError
from src.llm.mock import MockAdapter
from src.llm.registry import LLMRegistry
from src.observability.setup import setup_tracing, shutdown_tracing
from src.persistence.checkpointer import CheckpointerManager
from src.persistence.database import engine
from src.persistence.models.base import Base
from src.persistence.models.user import User
from src.persistence.models.session import Thread
from src.persistence.models.reminder import Reminder
from src.scheduler.setup import create_scheduler
from src.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("pagent starting up...")

    try:
        app.state.user_client = UserGrpcClient(settings.base_grpc_address)
    except Exception as e:
        logger.error(f"Failed to initialize UserGrpcClient: {e}")
        app.state.user_client = None

    try:
        app.state.checkpointer = CheckpointerManager(settings.postgres_url)
        await app.state.checkpointer.__aenter__()
    except Exception as e:
        logger.error(f"Failed to initialize CheckpointerManager: {e}")
        app.state.checkpointer = None

    # Ensure database exists on shared server
    try:
        from sqlalchemy import create_engine, text
        # Use a temporary sync engine to create database (connecting to 'mysql' system db)
        base_url = settings.mysql_url.split("/pagent")[0] + "/mysql"
        # Convert async driver to sync driver for this one-off task
        sync_url = base_url.replace("+aiomysql", "").replace("+asyncpg", "")
        temp_engine = create_engine(sync_url)
        with temp_engine.connect() as conn:
            conn.execute(text("CREATE DATABASE IF NOT EXISTS pagent"))
            conn.commit()
        temp_engine.dispose()
        logger.info("Database 'pagent' verified/created on shared server")
    except Exception as e:
        logger.warning(f"Failed to verify/create database 'pagent': {e}")

    # Create MySQL tables if they don't exist
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("MySQL tables verified/created")
    except Exception as e:
        logger.error(f"Failed to create MySQL tables: {e}")

    registry = LLMRegistry()
    mock = MockAdapter()
    registry.register(mock, primary=True)
    registry.register_as("default", mock)
    app.state.llm_registry = registry

    if settings.otel_enabled:
        try:
            setup_tracing(app)
        except Exception as e:
            logger.error(f"Failed to setup tracing: {e}")

    try:
        scheduler = create_scheduler()
        scheduler.start()
        app.state.scheduler = scheduler
        logger.info("Scheduler started with %d jobs", len(scheduler.get_jobs()))
    except Exception as e:
        logger.error(f"Failed to start scheduler: {e}")
        app.state.scheduler = None

    yield

    if app.state.scheduler:
        app.state.scheduler.shutdown(wait=False)
        logger.info("Scheduler shut down")
    
    if settings.otel_enabled:
        try:
            shutdown_tracing()
        except Exception as e:
            logger.error(f"Failed to shutdown tracing: {e}")

    if app.state.checkpointer:
        await app.state.checkpointer.__aexit__(None, None, None)
    
    if app.state.user_client:
        await app.state.user_client.close()
    
    await engine.dispose()
    logger.info("pagent shutting down...")


def create_app() -> FastAPI:
    app = FastAPI(
        title="pagent",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(LoggingMiddleware)
    app.add_middleware(
        RateLimitMiddleware,
        redis_url=settings.redis_url,
        limit=settings.quota_rate_limit_per_minute,
        window_seconds=60,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_exception_handler(AgentError, agent_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)

    app.include_router(health.router)
    app.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])
    app.include_router(sessions.router, prefix="/api/v1/sessions", tags=["sessions"])
    app.include_router(quota.router, prefix="/api/v1/quota", tags=["quota"])
    app.include_router(models.router, prefix="/api/v1/models", tags=["models"])
    app.include_router(agents.router, prefix="/api/v1/agents", tags=["agents"])
    app.include_router(metrics_router.router)

    return app


app = create_app()
