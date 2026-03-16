from __future__ import annotations

from typing import Any
from unittest import mock

from src.phoenix_tracing import (
    resolve_phoenix_tracing_settings,
    set_llm_input_messages,
    set_llm_output_message,
    set_llm_token_counts,
    set_retrieval_documents,
    set_span_attributes,
    start_span,
)


class _DummySpan:
    def __init__(self) -> None:
        self.attrs: dict[str, Any] = {}

    def set_attribute(self, key: str, value: Any) -> None:
        self.attrs[key] = value


def test_resolve_phoenix_disabled_by_default_without_env() -> None:
    with mock.patch.dict("os.environ", {}, clear=True):
        settings = resolve_phoenix_tracing_settings()

    assert settings.enabled is False
    assert settings.endpoint is None


def test_resolve_phoenix_enabled_when_endpoint_present() -> None:
    with mock.patch.dict(
        "os.environ",
        {"PHOENIX_COLLECTOR_ENDPOINT": "http://127.0.0.1:6006/v1/traces"},
        clear=True,
    ):
        settings = resolve_phoenix_tracing_settings()

    assert settings.enabled is True
    assert settings.endpoint == "http://127.0.0.1:6006/v1/traces"


def test_explicit_disable_overrides_env_auto_enable() -> None:
    with mock.patch.dict(
        "os.environ",
        {"PHOENIX_COLLECTOR_ENDPOINT": "http://127.0.0.1:6006/v1/traces"},
        clear=True,
    ):
        settings = resolve_phoenix_tracing_settings(enabled=False)

    assert settings.enabled is False


def test_set_span_attributes_serializes_complex_values() -> None:
    span = _DummySpan()

    set_span_attributes(
        span,
        {
            "simple": "ok",
            "primitive_list": ["a", "b", "c"],
            "complex": [{"id": 1, "score": 0.5}],
        },
    )

    assert span.attrs["simple"] == "ok"
    assert span.attrs["primitive_list"] == ["a", "b", "c"]
    assert isinstance(span.attrs["complex"], str)
    assert '"score": 0.5' in span.attrs["complex"]


def test_start_span_no_tracer_yields_none() -> None:
    with start_span(None, "test") as span:
        assert span is None


def test_set_retrieval_documents_emits_array_and_flattened_keys() -> None:
    span = _DummySpan()
    docs = [
        {
            "document.id": "chunk-1",
            "document.score": 0.91,
            "document.content": "hello",
        }
    ]

    set_retrieval_documents(span, docs)

    retrieval_docs = span.attrs.get("retrieval.documents")
    assert isinstance(retrieval_docs, list)
    assert retrieval_docs
    assert isinstance(retrieval_docs[0], str)
    assert '"document.id": "chunk-1"' in retrieval_docs[0]
    assert span.attrs["retrieval.documents.0.document.id"] == "chunk-1"
    assert span.attrs["retrieval.documents.0.document.score"] == 0.91


def test_set_llm_messages_and_token_counts_emit_flattened_fields() -> None:
    span = _DummySpan()
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is retrieval augmented generation?"},
    ]

    set_llm_input_messages(span, messages)
    set_llm_output_message(span, "RAG combines retrieval with generation.")
    set_llm_token_counts(span, prompt_tokens=123, completion_tokens=45, total_tokens=168)

    assert span.attrs["llm.input_messages.0.message.role"] == "system"
    assert span.attrs["llm.input_messages.1.message.role"] == "user"
    assert "retrieval augmented generation" in span.attrs["llm.input_messages.1.message.content"].lower()
    assert span.attrs["llm.output_messages.0.message.role"] == "assistant"
    assert "rag combines retrieval" in span.attrs["llm.output_messages.0.message.content"].lower()
    assert span.attrs["llm.token_count.prompt"] == 123
    assert span.attrs["llm.token_count.completion"] == 45
    assert span.attrs["llm.token_count.total"] == 168
