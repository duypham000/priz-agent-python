from collections.abc import AsyncIterator
import ssl

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.settings import settings


def _build_engine():
    url = settings.mysql_url
    connect_args = {}
    # aiomysql doesn't support ssl_ca/ssl_verify_cert/ssl_verify_identity as URL params.
    # Strip them and configure SSL via connect_args instead.
    tidb_ssl_params = ["ssl_ca", "ssl_verify_cert", "ssl_verify_identity"]
    if any(p in url for p in tidb_ssl_params):
        from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
        parsed = urlparse(url)
        qs = parse_qs(parsed.query, keep_blank_values=True)
        ca_path = qs.pop("ssl_ca", [None])[0]
        qs.pop("ssl_verify_cert", None)
        qs.pop("ssl_verify_identity", None)
        clean_url = urlunparse(parsed._replace(query=urlencode(qs, doseq=True)))
        ctx = ssl.create_default_context()
        if ca_path:
            try:
                ctx.load_verify_locations(ca_path)
            except Exception:
                ctx = ssl.create_default_context()
        connect_args["ssl"] = ctx
        return create_async_engine(clean_url, connect_args=connect_args, echo=False, pool_pre_ping=True)
    return create_async_engine(url, echo=False, pool_pre_ping=True)


engine = _build_engine()

AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_session() -> AsyncIterator[AsyncSession]:
    async with AsyncSessionLocal() as session:
        yield session
