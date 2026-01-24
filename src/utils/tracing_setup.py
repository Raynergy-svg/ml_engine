"""OpenTelemetry tracing setup for this repo.

Design goals:
- Safe no-op if OpenTelemetry deps are not installed.
- Idempotent: calling setup multiple times won't reconfigure global providers.
- Defaults to AI Toolkit collector: http://localhost:4318 (OTLP/HTTP).

Enable via env var:
- `ML_ENGINE_TRACING=1`
Optionally customize:
- `OTEL_SERVICE_NAME` (default: ml_engine)
- `OTEL_EXPORTER_OTLP_ENDPOINT` (default: http://localhost:4318)

This file intentionally avoids heavy auto-instrumentation; the training loop is
mostly in TensorFlow which lacks a stable upstream instrumentor.
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional


_lock = threading.Lock()
_tracing_initialized = False


def _env_truthy(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def tracing_enabled() -> bool:
    return _env_truthy(os.getenv("ML_ENGINE_TRACING"))


def setup_tracing(
    *,
    enabled: Optional[bool] = None,
    service_name: Optional[str] = None,
    exporter_endpoint: Optional[str] = None,
) -> bool:
    """Configure OpenTelemetry SDK + OTLP exporter.

    Returns True if tracing is active, else False.
    """
    global _tracing_initialized

    if enabled is None:
        enabled = tracing_enabled()
    if not enabled:
        return False

    def _normalize_otlp_http_endpoint(endpoint: str) -> str:
        endpoint = (endpoint or "").strip()
        if not endpoint:
            return "http://localhost:4318/v1/traces"
        # If user provided a base endpoint (common for AI Toolkit), make it a traces endpoint.
        # The OTLP HTTP trace exporter expects a full traces URL.
        if "/v1/traces" in endpoint:
            return endpoint
        # Strip trailing slashes then append standard path.
        endpoint = endpoint.rstrip("/")
        return f"{endpoint}/v1/traces"

    with _lock:
        if _tracing_initialized:
            return True

        try:
            from opentelemetry import trace
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
        except Exception:
            # Dependencies not installed (or incompatible). Don't crash training.
            return False

        service_name = service_name or os.getenv("OTEL_SERVICE_NAME") or "ml_engine"
        exporter_endpoint = (
            exporter_endpoint
            or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
            or "http://localhost:4318"
        )

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=_normalize_otlp_http_endpoint(exporter_endpoint))
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)

        _tracing_initialized = True
        return True


def get_tracer(name: str = "ml_engine"):
    try:
        from opentelemetry import trace

        return trace.get_tracer(name)
    except Exception:
        return None


@contextmanager
def span(
    name: str,
    *,
    attributes: Optional[Dict[str, Any]] = None,
) -> Iterator[None]:
    """Context manager creating a span when tracing is enabled."""
    tracer = get_tracer("ml_engine")
    if tracer is None:
        yield
        return

    try:
        with tracer.start_as_current_span(name) as sp:
            if attributes:
                for k, v in attributes.items():
                    try:
                        sp.set_attribute(k, v)
                    except Exception:
                        pass
            yield
    except Exception:
        # Never let tracing break training.
        yield


def add_event(name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
    try:
        from opentelemetry import trace

        sp = trace.get_current_span()
        if sp is None:
            return
        sp.add_event(name, attributes=attributes)
    except Exception:
        return


def record_exception(exc: BaseException) -> None:
    try:
        from opentelemetry import trace

        sp = trace.get_current_span()
        if sp is None:
            return
        sp.record_exception(exc)
    except Exception:
        return


def set_status_error(message: str) -> None:
    try:
        from opentelemetry import trace
        from opentelemetry.trace.status import Status, StatusCode

        sp = trace.get_current_span()
        if sp is None:
            return
        sp.set_status(Status(StatusCode.ERROR, message))
    except Exception:
        return


def gpu_probe_attributes() -> Dict[str, Any]:
    """Best-effort attributes for TF/Metal runtime visibility."""
    attrs: Dict[str, Any] = {}
    try:
        import tensorflow as tf

        attrs["tf.version"] = tf.__version__
        gpus = tf.config.list_physical_devices("GPU")
        attrs["tf.gpu.count"] = len(gpus)
        if gpus:
            attrs["tf.gpu.name"] = getattr(gpus[0], "name", "")
        # A small matmul forces device selection; if it errors it will be recorded.
        start = time.time()
        a = tf.random.uniform((1024, 1024))
        b = tf.random.uniform((1024, 1024))
        c = tf.linalg.matmul(a, b)
        _ = c.numpy()
        attrs["tf.matmul.ms"] = float((time.time() - start) * 1000.0)
    except Exception:
        pass
    return attrs
# — Raynergy-svg —
