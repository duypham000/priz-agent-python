"""OpenTelemetry tracing setup — exports to Arize Phoenix via OTLP HTTP."""

import logging

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from src.settings import settings

logger = logging.getLogger(__name__)

_tracer_provider: TracerProvider | None = None


def setup_tracing(app=None) -> TracerProvider:
    """Initialize tracing. Call once during app startup."""
    global _tracer_provider

    resource = Resource.create({SERVICE_NAME: settings.otel_service_name})
    provider = TracerProvider(resource=resource)

    otlp = OTLPSpanExporter(endpoint=f"{settings.phoenix_endpoint}/v1/traces")
    provider.add_span_processor(BatchSpanProcessor(otlp))

    trace.set_tracer_provider(provider)
    _tracer_provider = provider

    if app is not None:
        FastAPIInstrumentor().instrument_app(app)

    logger.info("Tracing → %s", settings.phoenix_endpoint)
    return provider


def get_tracer(name: str = "pagent") -> trace.Tracer:
    return trace.get_tracer(name)


def shutdown_tracing() -> None:
    if _tracer_provider is not None:
        _tracer_provider.shutdown()
