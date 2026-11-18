"""AgentLens: OpenTelemetry tracing and metrics for LangGraph / LangChain agents.

    import agentlens
    agentlens.configure(service_name="support-bot")      # OTLP -> your collector
    app = agentlens.instrument(graph.compile(), agent_name="support-bot")
    app.invoke({"messages": [...]})
"""

from __future__ import annotations

import os
from typing import Any

from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.metrics.view import ExplicitBucketHistogramAggregation, View
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from . import semconv
from .handler import AgentLensCallbackHandler
from .pricing import cost_usd, register_price

__all__ = ["configure", "instrument", "callback_handler", "shutdown", "AgentLensCallbackHandler",
           "register_price", "cost_usd", "semconv"]
__version__ = "0.1.0"

_DURATION_BUCKETS = [0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 15, 30, 60, 120]
_TOKEN_BUCKETS = [16, 64, 256, 1024, 2048, 4096, 8192, 16384, 65536, 262144]
_COST_BUCKETS = [0.00001, 0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5, 1]

_tracer_provider: TracerProvider | None = None
_meter_provider: MeterProvider | None = None


def _views() -> list[View]:
    """Bucket boundaries sized for LLM work (the OTel defaults suit milliseconds)."""
    return [
        View(instrument_name=semconv.M_OP_DURATION, aggregation=ExplicitBucketHistogramAggregation(_DURATION_BUCKETS)),
        View(instrument_name=semconv.M_NODE_DURATION, aggregation=ExplicitBucketHistogramAggregation(_DURATION_BUCKETS)),
        View(instrument_name=semconv.M_TOOL_DURATION, aggregation=ExplicitBucketHistogramAggregation(_DURATION_BUCKETS)),
        View(instrument_name=semconv.M_RUN_DURATION, aggregation=ExplicitBucketHistogramAggregation(_DURATION_BUCKETS)),
        View(instrument_name=semconv.M_TOKEN_USAGE, aggregation=ExplicitBucketHistogramAggregation(_TOKEN_BUCKETS)),
        View(instrument_name=semconv.M_RUN_COST, aggregation=ExplicitBucketHistogramAggregation(_COST_BUCKETS)),
    ]


def configure(service_name: str = "agent", endpoint: str | None = None,
              resource_attributes: dict[str, Any] | None = None,
              metric_interval_ms: int | None = None) -> None:
    """Set global OTel tracer and meter providers that export over OTLP/HTTP.

    ``endpoint`` defaults to ``OTEL_EXPORTER_OTLP_ENDPOINT`` or http://localhost:4318.
    Skip this if your app already configures OpenTelemetry: AgentLens uses
    whatever global providers are set.
    """
    global _tracer_provider, _meter_provider
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    base = (endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or "http://localhost:4318").rstrip("/")
    resource = Resource.create({"service.name": service_name, **(resource_attributes or {})})

    _tracer_provider = TracerProvider(resource=resource)
    _tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{base}/v1/traces")))
    trace.set_tracer_provider(_tracer_provider)

    interval = metric_interval_ms or int(os.getenv("OTEL_METRIC_EXPORT_INTERVAL", "10000"))
    reader = PeriodicExportingMetricReader(OTLPMetricExporter(endpoint=f"{base}/v1/metrics"),
                                           export_interval_millis=interval)
    _meter_provider = MeterProvider(resource=resource, metric_readers=[reader], views=_views())
    metrics.set_meter_provider(_meter_provider)


def callback_handler(agent_name: str = "agent", capture_content: bool | None = None,
                     **kwargs: Any) -> AgentLensCallbackHandler:
    """A handler to pass yourself: ``app.invoke(x, config={"callbacks": [handler]})``."""
    if capture_content is None:
        capture_content = os.getenv("AGENTLENS_CAPTURE_CONTENT", "false").lower() in ("1", "true", "yes")
    return AgentLensCallbackHandler(agent_name=agent_name, capture_content=capture_content, **kwargs)


def instrument(runnable: Any, agent_name: str | None = None, capture_content: bool | None = None,
               **kwargs: Any) -> Any:
    """Return the graph/runnable with AgentLens attached to every invocation.

    Works with compiled LangGraph graphs and any LangChain Runnable; the result
    supports invoke/ainvoke/stream/astream/batch like the original.
    """
    name = agent_name or getattr(runnable, "name", None) or "agent"
    handler = callback_handler(name, capture_content, **kwargs)
    return runnable.with_config({"callbacks": [handler]})


def shutdown() -> None:
    """Flush and stop the providers created by ``configure()``. Call before exit."""
    if _tracer_provider:
        _tracer_provider.shutdown()
    if _meter_provider:
        _meter_provider.shutdown()
