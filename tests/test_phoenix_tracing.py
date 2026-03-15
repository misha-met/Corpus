from __future__ import annotations

from typing import Any
from unittest import mock

from src.phoenix_tracing import (
    resolve_phoenix_tracing_settings,
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
