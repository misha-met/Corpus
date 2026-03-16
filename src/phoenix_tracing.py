"""Phoenix tracing integration helpers.

This module centralizes Arize Phoenix/OpenTelemetry setup and provides
safe span helper utilities used by the RAG pipeline.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Generator, Optional

logger = logging.getLogger(__name__)

OPENINFERENCE_SPAN_KIND = "openinference.span.kind"
SPAN_KIND_CHAIN = "CHAIN"
SPAN_KIND_RETRIEVER = "RETRIEVER"
SPAN_KIND_RERANKER = "RERANKER"
SPAN_KIND_LLM = "LLM"
SPAN_KIND_EMBEDDING = "EMBEDDING"
SPAN_KIND_TOOL = "TOOL"


@dataclass(frozen=True)
class PhoenixTracingSettings:
    """Effective Phoenix tracing settings after env/override resolution."""

    enabled: bool
    project_name: str
    endpoint: Optional[str]
    api_key: Optional[str]
    auto_instrument: bool
    batch: bool


@dataclass(frozen=True)
class PhoenixTracingStatus:
    """Runtime status for Phoenix tracing setup."""

    configured: bool
    active: bool
    project_name: Optional[str] = None
    endpoint: Optional[str] = None
    error: Optional[str] = None


_RUNTIME_LOCK = threading.Lock()
_RUNTIME_PROVIDER: Any = None
_RUNTIME_CONFIG_KEY: Optional[tuple[str, str, str, bool, bool]] = None
_RUNTIME_STATUS = PhoenixTracingStatus(configured=False, active=False)


_TRUE_VALUES = {"1", "true", "yes", "on", "y"}
_FALSE_VALUES = {"0", "false", "no", "off", "n"}


def _coerce_bool(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if not normalized:
        return default
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    return default


def resolve_phoenix_tracing_settings(
    *,
    enabled: Optional[bool] = None,
    project_name: Optional[str] = None,
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    auto_instrument: Optional[bool] = None,
    batch: Optional[bool] = None,
) -> PhoenixTracingSettings:
    """Resolve tracing settings from explicit overrides and environment vars."""

    env_enabled_raw = os.getenv("RAG_PHOENIX_ENABLED")
    env_project = os.getenv("PHOENIX_PROJECT_NAME")
    env_endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
    env_api_key = os.getenv("PHOENIX_API_KEY")
    env_auto_instrument_raw = os.getenv("RAG_PHOENIX_AUTO_INSTRUMENT")
    env_batch_raw = os.getenv("RAG_PHOENIX_BATCH")

    if enabled is not None:
        resolved_enabled = bool(enabled)
    elif env_enabled_raw is not None:
        resolved_enabled = _coerce_bool(env_enabled_raw, default=False)
    else:
        # If endpoint/api key are provided, assume tracing should be enabled.
        resolved_enabled = bool((env_endpoint or "").strip() or (env_api_key or "").strip())

    resolved_project_name = (
        (project_name or "").strip()
        or (env_project or "").strip()
        or "dh-notebook-offline"
    )

    resolved_endpoint = (endpoint or "").strip() or (env_endpoint or "").strip() or None
    if resolved_enabled and resolved_endpoint is None:
        # Local Phoenix default.
        resolved_endpoint = "http://127.0.0.1:6006/v1/traces"

    resolved_api_key = (api_key or "").strip() or (env_api_key or "").strip() or None

    if auto_instrument is not None:
        resolved_auto_instrument = bool(auto_instrument)
    else:
        resolved_auto_instrument = _coerce_bool(env_auto_instrument_raw, default=False)

    if batch is not None:
        resolved_batch = bool(batch)
    else:
        resolved_batch = _coerce_bool(env_batch_raw, default=True)

    return PhoenixTracingSettings(
        enabled=resolved_enabled,
        project_name=resolved_project_name,
        endpoint=resolved_endpoint,
        api_key=resolved_api_key,
        auto_instrument=resolved_auto_instrument,
        batch=resolved_batch,
    )


def _config_key(settings: PhoenixTracingSettings) -> tuple[str, str, str, bool, bool]:
    return (
        settings.project_name,
        settings.endpoint or "",
        "***" if settings.api_key else "",
        settings.auto_instrument,
        settings.batch,
    )


def configure_phoenix_tracing(
    *,
    enabled: Optional[bool] = None,
    project_name: Optional[str] = None,
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    auto_instrument: Optional[bool] = None,
    batch: Optional[bool] = None,
) -> tuple[Any, PhoenixTracingStatus]:
    """Configure Phoenix tracing once and return tracer provider + status."""

    global _RUNTIME_PROVIDER, _RUNTIME_CONFIG_KEY, _RUNTIME_STATUS

    settings = resolve_phoenix_tracing_settings(
        enabled=enabled,
        project_name=project_name,
        endpoint=endpoint,
        api_key=api_key,
        auto_instrument=auto_instrument,
        batch=batch,
    )

    if not settings.enabled:
        status = PhoenixTracingStatus(
            configured=False,
            active=False,
            project_name=settings.project_name,
            endpoint=settings.endpoint,
        )
        with _RUNTIME_LOCK:
            _RUNTIME_STATUS = status
        return None, status

    key = _config_key(settings)

    with _RUNTIME_LOCK:
        if _RUNTIME_PROVIDER is not None and _RUNTIME_CONFIG_KEY == key:
            return _RUNTIME_PROVIDER, _RUNTIME_STATUS

        if _RUNTIME_PROVIDER is not None and _RUNTIME_CONFIG_KEY != key:
            logger.warning(
                "Phoenix tracing already initialized with a different configuration; reusing active provider"
            )
            return _RUNTIME_PROVIDER, _RUNTIME_STATUS

        try:
            from phoenix.otel import register
        except Exception as exc:  # pragma: no cover - optional dependency
            status = PhoenixTracingStatus(
                configured=True,
                active=False,
                project_name=settings.project_name,
                endpoint=settings.endpoint,
                error=(
                    "Phoenix tracing requested but arize-phoenix-otel is unavailable: "
                    f"{type(exc).__name__}: {exc}"
                ),
            )
            _RUNTIME_STATUS = status
            logger.warning(status.error)
            return None, status

        kwargs: dict[str, Any] = {
            "project_name": settings.project_name,
            "auto_instrument": settings.auto_instrument,
            "batch": settings.batch,
        }
        if settings.endpoint:
            kwargs["endpoint"] = settings.endpoint
        if settings.api_key:
            kwargs["api_key"] = settings.api_key

        try:
            provider = register(**kwargs)
        except Exception as exc:  # pragma: no cover - runtime dependency
            status = PhoenixTracingStatus(
                configured=True,
                active=False,
                project_name=settings.project_name,
                endpoint=settings.endpoint,
                error=(
                    "Phoenix tracing setup failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
            )
            _RUNTIME_STATUS = status
            logger.warning(status.error)
            return None, status

        _RUNTIME_PROVIDER = provider
        _RUNTIME_CONFIG_KEY = key
        _RUNTIME_STATUS = PhoenixTracingStatus(
            configured=True,
            active=True,
            project_name=settings.project_name,
            endpoint=settings.endpoint,
        )
        logger.info(
            "Phoenix tracing enabled (project=%s endpoint=%s)",
            settings.project_name,
            settings.endpoint,
        )
        return _RUNTIME_PROVIDER, _RUNTIME_STATUS


def get_phoenix_tracer(
    scope_name: str,
    *,
    enabled: Optional[bool] = None,
    project_name: Optional[str] = None,
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    auto_instrument: Optional[bool] = None,
    batch: Optional[bool] = None,
) -> tuple[Any, PhoenixTracingStatus]:
    """Return a configured tracer for the requested instrumentation scope."""

    provider, status = configure_phoenix_tracing(
        enabled=enabled,
        project_name=project_name,
        endpoint=endpoint,
        api_key=api_key,
        auto_instrument=auto_instrument,
        batch=batch,
    )
    if provider is None:
        return None, status
    try:
        return provider.get_tracer(scope_name), status
    except Exception as exc:  # pragma: no cover - runtime dependency
        fallback = PhoenixTracingStatus(
            configured=True,
            active=False,
            project_name=status.project_name,
            endpoint=status.endpoint,
            error=f"Failed to create tracer '{scope_name}': {type(exc).__name__}: {exc}",
        )
        return None, fallback


def get_phoenix_runtime_status() -> PhoenixTracingStatus:
    """Return the current runtime status without forcing initialization."""

    with _RUNTIME_LOCK:
        if _RUNTIME_PROVIDER is not None or _RUNTIME_STATUS.configured:
            return _RUNTIME_STATUS

    settings = resolve_phoenix_tracing_settings()
    if not settings.enabled:
        return PhoenixTracingStatus(
            configured=False,
            active=False,
            project_name=settings.project_name,
            endpoint=settings.endpoint,
        )

    return PhoenixTracingStatus(
        configured=True,
        active=False,
        project_name=settings.project_name,
        endpoint=settings.endpoint,
        error="Tracing configured but not initialized yet",
    )


def _normalize_attr_value(value: Any, *, max_text_chars: int) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, str):
        if len(value) <= max_text_chars:
            return value
        return value[:max_text_chars] + "..."
    if isinstance(value, (list, tuple)):
        if all(isinstance(item, (str, bool, int, float)) for item in value):
            normalized: list[Any] = []
            for item in value:
                if isinstance(item, str) and len(item) > max_text_chars:
                    normalized.append(item[:max_text_chars] + "...")
                else:
                    normalized.append(item)
            return normalized

    try:
        serialized = json.dumps(value, ensure_ascii=True)
    except (TypeError, ValueError):
        try:
            serialized = str(value)
        except Exception:
            logger.debug("Failed to stringify attribute value of type %s", type(value).__name__, exc_info=True)
            return None

    if len(serialized) > max_text_chars:
        return serialized[:max_text_chars] + "..."
    return serialized


def set_span_attribute(span: Any, key: str, value: Any, *, max_text_chars: int = 4096) -> None:
    """Set one span attribute safely, guarding unsupported value shapes."""

    if span is None:
        return
    normalized = _normalize_attr_value(value, max_text_chars=max_text_chars)
    if normalized is None:
        return
    try:
        span.set_attribute(key, normalized)
    except Exception:
        logger.debug("Failed to set span attribute %s", key, exc_info=True)


def set_span_attributes(span: Any, attributes: dict[str, Any], *, max_text_chars: int = 4096) -> None:
    """Set multiple span attributes safely."""

    if span is None:
        return
    for key, value in attributes.items():
        set_span_attribute(span, key, value, max_text_chars=max_text_chars)


def format_openinference_document(
    document_id: str,
    content: str,
    score: float,
    metadata: Optional[dict[str, Any]] = None,
    max_content_chars: int = 400,
) -> dict[str, Any]:
    """Build a single OpenInference-style document dictionary.

    Consolidates the logic for mapping retrieval results to the schema
    expected by Phoenix and other OpenInference-compatible UIs.
    """

    clean_content = content.strip()
    if len(clean_content) > max_content_chars:
        clean_content = clean_content[:max_content_chars] + "..."

    meta = metadata if isinstance(metadata, dict) else {}

    return {
        "document.id": document_id,
        "document.score": round(float(score), 6),
        "document.content": clean_content,
        "document.metadata": {
            "source_id": meta.get("source_id"),
            "page_number": meta.get("page_number"),
            "display_page": meta.get("display_page"),
            "header_path": meta.get("header_path"),
        },
    }


def set_retrieval_documents(
    span: Any,
    documents: list[dict[str, Any]],
    *,
    max_text_chars: int = 4096,
) -> None:
    """Set retrieval documents using canonical and flattened OpenInference keys."""

    if span is None:
        return

    docs = documents if isinstance(documents, list) else []
    # OpenTelemetry attributes cannot carry nested object arrays directly.
    # Keep the canonical key as a string array so UI code that expects an array
    # does not crash, and provide flattened keys for structured inspection.
    docs_json = [
        to_json(doc) if isinstance(doc, dict) else str(doc)
        for doc in docs
    ]
    set_span_attribute(span, "retrieval.documents", docs_json, max_text_chars=max_text_chars)

    # Also emit flattened keys (retrieval.documents.{i}.document.*) for UIs that
    # rely on OpenInference flattening patterns.
    for index, doc in enumerate(docs):
        if not isinstance(doc, dict):
            continue
        prefix = f"retrieval.documents.{index}"
        for key, value in doc.items():
            set_span_attribute(span, f"{prefix}.{key}", value, max_text_chars=max_text_chars)


def set_reranker_documents(
    span: Any,
    *,
    input_documents: list[dict[str, Any]],
    output_documents: list[dict[str, Any]],
    query: Optional[str] = None,
    top_k: Optional[int] = None,
    max_text_chars: int = 4096,
) -> None:
    """Emit OpenInference RERANKER attributes: input/output documents, query, top_k."""

    if span is None:
        return

    if query is not None:
        set_span_attribute(span, "reranker.query", query, max_text_chars=max_text_chars)
    if top_k is not None:
        set_span_attribute(span, "reranker.top_k", int(top_k))

    # Emit root array for UI parsing
    if input_documents:
        docs_json = [to_json(doc) if isinstance(doc, dict) else str(doc) for doc in input_documents]
        set_span_attribute(span, "reranker.input_documents", docs_json, max_text_chars=max_text_chars)

    # Emit flattened reranker.input_documents.{i}.document.* keys
    for index, doc in enumerate(input_documents):
        if not isinstance(doc, dict):
            continue
        prefix = f"reranker.input_documents.{index}"
        for key, value in doc.items():
            set_span_attribute(span, f"{prefix}.{key}", value, max_text_chars=max_text_chars)

    # Emit root array for UI parsing
    if output_documents:
        docs_json = [to_json(doc) if isinstance(doc, dict) else str(doc) for doc in output_documents]
        set_span_attribute(span, "reranker.output_documents", docs_json, max_text_chars=max_text_chars)

    # Emit flattened reranker.output_documents.{i}.document.* keys
    for index, doc in enumerate(output_documents):
        if not isinstance(doc, dict):
            continue
        prefix = f"reranker.output_documents.{index}"
        for key, value in doc.items():
            set_span_attribute(span, f"{prefix}.{key}", value, max_text_chars=max_text_chars)


def set_llm_input_messages(
    span: Any,
    messages: list[dict[str, Any]],
    *,
    max_messages: int = 12,
    max_text_chars: int = 4096,
) -> None:
    """Emit flattened OpenInference `llm.input_messages.*` attributes."""

    if span is None:
        return

    for index, message in enumerate(messages[: max(1, max_messages)]):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "user"))
        content = str(message.get("content", ""))
        set_span_attribute(span, f"llm.input_messages.{index}.message.role", role)
        set_span_attribute(
            span,
            f"llm.input_messages.{index}.message.content",
            content,
            max_text_chars=max_text_chars,
        )


def set_llm_output_message(
    span: Any,
    content: str,
    *,
    role: str = "assistant",
    max_text_chars: int = 4096,
) -> None:
    """Emit flattened OpenInference `llm.output_messages.0.*` attributes."""

    if span is None:
        return
    set_span_attribute(span, "llm.output_messages.0.message.role", role)
    set_span_attribute(
        span,
        "llm.output_messages.0.message.content",
        content,
        max_text_chars=max_text_chars,
    )


def set_llm_token_counts(
    span: Any,
    *,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
) -> None:
    """Emit OpenInference `llm.token_count.*` attributes."""

    if span is None:
        return
    if prompt_tokens is not None:
        set_span_attribute(span, "llm.token_count.prompt", int(prompt_tokens))
    if completion_tokens is not None:
        set_span_attribute(span, "llm.token_count.completion", int(completion_tokens))
    if total_tokens is not None:
        set_span_attribute(span, "llm.token_count.total", int(total_tokens))


@contextmanager
def start_span(
    tracer: Any,
    name: str,
    *,
    span_kind: Optional[str] = None,
    attributes: Optional[dict[str, Any]] = None,
) -> Generator[Any, None, None]:
    """Create a span if tracing is enabled; otherwise yield ``None``."""

    if tracer is None:
        yield None
        return

    with tracer.start_as_current_span(name) as span:
        if span_kind:
            set_span_attribute(span, OPENINFERENCE_SPAN_KIND, span_kind)
        if attributes:
            set_span_attributes(span, attributes)
        yield span


def mark_span_error(span: Any, message: str) -> None:
    """Best-effort error annotation for spans."""

    if span is None:
        return

    set_span_attribute(span, "error", True)
    set_span_attribute(span, "error.message", message)
    try:  # pragma: no cover - optional runtime dependency
        from opentelemetry.trace import Status, StatusCode

        span.set_status(Status(StatusCode.ERROR, message))
    except Exception:
        pass


def to_json(value: Any) -> str:
    """JSON helper for span attributes and structured diagnostics."""

    return json.dumps(value, ensure_ascii=True)


def annotate_span_feedback(
    *,
    span_id: str,
    trace_id: str,
    label: str,
    score: Optional[float] = None,
    comment: Optional[str] = None,
    annotator: str = "user",
) -> bool:
    """Attach a user feedback annotation to a span in Phoenix.

    Requires ``arize-phoenix`` (not just ``arize-phoenix-otel``).
    Returns True on success, False if the package is unavailable or the call fails.
    """
    if not trace_id or not span_id:
        logger.warning("annotate_span_feedback requires both trace_id and span_id")
        return False

    try:
        import phoenix as px  # type: ignore[import-untyped]
        from phoenix.trace.span_evaluations import SpanEvaluations  # type: ignore[import-untyped]
    except ImportError:
        logger.debug("arize-phoenix not installed — cannot annotate spans")
        return False

    try:
        from urllib.parse import urlparse

        settings = resolve_phoenix_tracing_settings()
        base_endpoint = "http://127.0.0.1:6006"
        ep = settings.endpoint
        if isinstance(ep, str):
            try:
                parsed = urlparse(ep)
                if parsed.scheme and parsed.netloc:
                    base_endpoint = f"{parsed.scheme}://{parsed.netloc}"
            except Exception:
                pass

        client = px.Client(endpoint=base_endpoint)
        import pandas as pd  # type: ignore[import-untyped]

        df = pd.DataFrame(
            [
                {
                    "span_id": span_id,
                    "trace_id": trace_id,
                    "label": label,
                    "score": score,
                    "explanation": comment or "",
                }
            ]
        ).set_index("span_id")
        
        evals = SpanEvaluations(eval_name="user-feedback", dataframe=df)
        client.log_evaluations(evals)
        logger.info("Phoenix annotation logged: trace_id=%s span_id=%s label=%s", trace_id, span_id, label)
        return True
    except Exception as exc:
        logger.warning("Failed to annotate span in Phoenix: %s", exc)
        return False
