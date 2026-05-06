import os
import sys
from logging.config import fileConfig
from pathlib import Path

# Ensure project root is on sys.path so `src` is importable
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from alembic import context
from sqlalchemy import engine_from_config, pool

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set URL from environment (overrides alembic.ini placeholder)
mysql_url = os.environ.get("MYSQL_URL", "")
# Alembic needs a sync URL — strip async driver prefix
sync_url = mysql_url.replace("mysql+aiomysql://", "mysql+pymysql://")
config.set_main_option("sqlalchemy.url", sync_url)

from src.persistence.models.base import Base  # noqa: E402  (must be after sys.path setup)
from src.persistence.models.quota import TokenUsage  # noqa: F401 (register with Base.metadata)
from src.persistence.models.session import Thread  # noqa: F401
from src.persistence.models.user import User  # noqa: F401
from src.persistence.models.reminder import Reminder  # noqa: F401

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
