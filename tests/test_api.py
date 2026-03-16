"""Tests for the FastAPI chat endpoint and health route.

Uses httpx.AsyncClient against the real FastAPI app with a mocked
RagEngine that yields query events, avoiding ML model loading.
"""

from __future__ import annotations

import json
from typing import Optional
from unittest.mock import patch

import httpx
import pytest

from src.api import app
from src.query_events import (
    ErrorEvent,
    FinishEvent,
    IntentEvent,
    SourcesEvent,
    StatusEvent,
    TextTokenEvent,
)


# ---------------------------------------------------------------------------
# Mock RagEngine with query_events support
# ---------------------------------------------------------------------------


class MockRagEngine:
    """Mock RagEngine that yields query events without loading ML models."""

    def __init__(
        self,
        *,
        answer: str = "This is a test answer from the RAG engine.",
        intent: str = "analyze",
        confidence: float = 0.85,
        method: str = "heuristic",
        source_ids: Optional[list[str]] = None,
        fail: bool = False,
        fail_during_generation: bool = False,
    ):
        self._answer = answer
        self._intent = intent
        self._confidence = confidence
        self._method = method
        self._source_ids = source_ids or ["test_source"]
        self._fail = fail
        self._fail_during_generation = fail_during_generation
        self.query_count = 0

    def query_events(self, query_text, *, source_id=None, intent_override=None, citations_enabled=None, should_stop=None, enable_thinking=None):
        """Yield mock query events."""
        self.query_count += 1
        _stop = should_stop or (lambda: False)

        if self._fail:
            raise RuntimeError("Mock engine failure")

        yield StatusEvent(status="Preparing retrieval models...")
        if _stop():
            yield ErrorEvent(code="STREAM_CANCELLED", message="Cancelled")
            yield FinishEvent(finish_reason="error")
            return

        yield StatusEvent(status="Classifying intent...")
        yield IntentEvent(
            intent=self._intent,
            confidence=self._confidence,
            method=self._method,
        )

        if _stop():
            yield ErrorEvent(code="STREAM_CANCELLED", message="Cancelled")
            yield FinishEvent(finish_reason="error")
            return

        yield StatusEvent(status="Searching knowledge base...")

        source_ids = [source_id] if source_id else self._source_ids
        yield SourcesEvent(source_ids=source_ids)

        yield StatusEvent(status="Generating answer...")

        if self._fail_during_generation:
            yield ErrorEvent(code="INTERNAL", message="Generation failed mid-stream")
            yield FinishEvent(finish_reason="error")
            return

        # Stream answer as individual tokens (like real generator)
        answer = f"Answer to: {query_text}" if "Answer" not in self._answer else self._answer
        words = answer.split(" ")
        token_count = 0
        for i, word in enumerate(words):
            if _stop():
                yield ErrorEvent(code="STREAM_CANCELLED", message="Cancelled during generation")
                yield FinishEvent(finish_reason="error")
                return
            token = word if i == 0 else f" {word}"
            yield TextTokenEvent(token=token)
            token_count += 1

        yield FinishEvent(finish_reason="stop", completion_tokens=token_count)

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_sse_events(body: str) -> list[dict]:
    """Split a full SSE response body into parsed event dicts.

    Skips the terminal ``data: [DONE]`` sentinel.
    """
    events: list[dict] = []
    for chunk in body.split("\n\n"):
        chunk = chunk.strip()
        if not chunk or chunk == "data: [DONE]":
            continue
        assert chunk.startswith("data: "), f"Unexpected SSE chunk: {chunk!r}"
        events.append(json.loads(chunk[6:]))
    return events


def _get_sse_text(events: list[dict]) -> str:
    """Concatenate all text-delta deltas from the event list."""
    return "".join(e["delta"] for e in events if e.get("type") == "text-delta")


def _get_sse_annotations_by_type(events: list[dict], ann_type: str) -> list[dict]:
    """Return data payloads from data-{ann_type} frames."""
    key = f"data-{ann_type}"
    return [e["data"] for e in events if e.get("type") == key]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_engine():
    return MockRagEngine()


@pytest.fixture
def mock_engine_failing():
    return MockRagEngine(fail=True)


@pytest.fixture
def mock_engine_gen_fail():
    return MockRagEngine(fail_during_generation=True)


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    @pytest.mark.anyio
    async def test_health_returns_ok(self) -> None:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"


# ---------------------------------------------------------------------------
# Chat endpoint — happy path
# ---------------------------------------------------------------------------


class TestChatHappyPath:
    @pytest.mark.anyio
    async def test_stream_contains_text_and_finish(self, mock_engine) -> None:
        """Valid chat request returns SSE stream with a text answer and finish frame."""
        with patch("src.api._get_engine", return_value=mock_engine):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/api/chat",
                    json={"messages": [{"role": "user", "content": "What is this?"}]},
                )

        assert resp.status_code == 200
        events = _parse_sse_events(resp.text)
        assert _get_sse_text(events) == "Answer to: What is this?"
        assert any(e.get("type") == "finish" for e in events)

    @pytest.mark.anyio
    async def test_stream_headers(self, mock_engine) -> None:
        """Response has correct SSE headers."""
        with patch("src.api._get_engine", return_value=mock_engine):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/api/chat",
                    json={"messages": [{"role": "user", "content": "Hello"}]},
                )

        assert resp.headers["content-type"] == "text/event-stream; charset=utf-8"
        assert resp.headers["cache-control"] == "no-cache"
        assert resp.headers["x-accel-buffering"] == "no"

    @pytest.mark.anyio
    async def test_stream_contains_intent_annotation(self, mock_engine) -> None:
        """SSE stream includes an intent annotation frame."""
        with patch("src.api._get_engine", return_value=mock_engine):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/api/chat",
                    json={"messages": [{"role": "user", "content": "Analyze this"}]},
                )

        assert resp.status_code == 200
        events = _parse_sse_events(resp.text)
        intent_anns = _get_sse_annotations_by_type(events, "intent")
        assert len(intent_anns) == 1
        assert intent_anns[0]["intent"] == "analyze"
        assert intent_anns[0]["confidence"] == 0.85

    @pytest.mark.anyio
    async def test_stream_contains_source_annotation(self, mock_engine) -> None:
        """SSE stream includes a sources annotation frame."""
        with patch("src.api._get_engine", return_value=mock_engine):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/api/chat",
                    json={"messages": [{"role": "user", "content": "Query"}]},
                )

        assert resp.status_code == 200
        events = _parse_sse_events(resp.text)
        source_anns = _get_sse_annotations_by_type(events, "sources")
        assert len(source_anns) == 1
        assert "test_source" in source_anns[0]["sourceIds"]

    @pytest.mark.anyio
    async def test_stream_contains_status_annotations(self, mock_engine) -> None:
        """SSE stream includes multiple status annotation frames."""
        with patch("src.api._get_engine", return_value=mock_engine):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/api/chat",
                    json={"messages": [{"role": "user", "content": "Question"}]},
                )

        assert resp.status_code == 200
        events = _parse_sse_events(resp.text)
        status_anns = _get_sse_annotations_by_type(events, "status")
        assert len(status_anns) >= 3  # Loading RAG engine + Classifying + Searching + Generating

    @pytest.mark.anyio
    async def test_source_filter_passed_through(self, mock_engine) -> None:
        """Optional source_id from data field is passed to engine; stream returns 200 and answer."""
        with patch("src.api._get_engine", return_value=mock_engine):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/api/chat",
                    json={
                        "messages": [{"role": "user", "content": "Query"}],
                        "data": {"source_id": "specific_doc"},
                    },
                )

        assert resp.status_code == 200
        events = _parse_sse_events(resp.text)
        # source_id "specific_doc" is threaded through; check sources annotation
        source_anns = _get_sse_annotations_by_type(events, "sources")
        assert any("specific_doc" in a["sourceIds"] for a in source_anns)

    @pytest.mark.anyio
    async def test_multi_source_ids_do_not_collapse_to_first_source(self, mock_engine) -> None:
        """When multiple source_ids are provided, backend should not pin to source_ids[0]."""
        with patch("src.api._get_engine", return_value=mock_engine):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/api/chat",
                    json={
                        "messages": [{"role": "user", "content": "Query"}],
                        "data": {"source_ids": ["doc_a", "doc_b"]},
                    },
                )

        assert resp.status_code == 200
        events = _parse_sse_events(resp.text)
        source_anns = _get_sse_annotations_by_type(events, "sources")
        assert len(source_anns) >= 1
        # Mock engine emits "test_source" when source pinning is not applied.
        assert any("test_source" in a["sourceIds"] for a in source_anns)

    @pytest.mark.anyio
    async def test_chat_with_history(self, mock_engine) -> None:
        """Chat with multiple messages works (last message is the query)."""
        with patch("src.api._get_engine", return_value=mock_engine):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/api/chat",
                    json={
                        "messages": [
                            {"role": "user", "content": "First question"},
                            {"role": "assistant", "content": "First answer"},
                            {"role": "user", "content": "Follow up"},
                        ]
                    },
                )

        assert resp.status_code == 200
        events = _parse_sse_events(resp.text)
        assert _get_sse_text(events) == "Answer to: Follow up"

    @pytest.mark.anyio
    async def test_text_tokens_arrive_individually(self, mock_engine) -> None:
        """SSE stream contains multiple text-delta frames concatenating into the full answer."""
        with patch("src.api._get_engine", return_value=mock_engine):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/api/chat",
                    json={"messages": [{"role": "user", "content": "Test"}]},
                )

        events = _parse_sse_events(resp.text)
        text_deltas = [e for e in events if e.get("type") == "text-delta"]
        assert len(text_deltas) > 1, "Expected multiple text-delta frames"
        assert _get_sse_text(events) == "Answer to: Test"


# ---------------------------------------------------------------------------
# Chat endpoint — validation errors
# ---------------------------------------------------------------------------


class TestChatValidation:
    @pytest.mark.anyio
    async def test_empty_messages_rejected(self) -> None:
        """Empty messages array returns 422."""
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/api/chat", json={"messages": []})
        assert resp.status_code == 422

    @pytest.mark.anyio
    async def test_missing_messages_rejected(self) -> None:
        """Missing messages field returns 422."""
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/api/chat", json={})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Chat endpoint — error handling
# ---------------------------------------------------------------------------


class TestChatEngineErrors:
    @pytest.mark.anyio
    async def test_engine_exception_produces_error_stream(
        self, mock_engine_failing
    ) -> None:
        """Engine exception produces structured error SSE frame with INTERNAL code."""
        with patch("src.api._get_engine", return_value=mock_engine_failing):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/api/chat",
                    json={"messages": [{"role": "user", "content": "Will fail"}]},
                )

        assert resp.status_code == 200
        events = _parse_sse_events(resp.text)
        error_anns = _get_sse_annotations_by_type(events, "error")
        assert any(a["error"]["code"] == "INTERNAL" for a in error_anns)
        stream_errors = [e for e in events if e.get("type") == "error"]
        assert any("Mock engine failure" in e["error"] for e in stream_errors)

    @pytest.mark.anyio
    async def test_mid_generation_error_produces_error_stream(
        self, mock_engine_gen_fail
    ) -> None:
        """Error during generation (after status events) still produces proper error SSE frames."""
        with patch("src.api._get_engine", return_value=mock_engine_gen_fail):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/api/chat",
                    json={"messages": [{"role": "user", "content": "Generate fail"}]},
                )

        events = _parse_sse_events(resp.text)
        error_anns = _get_sse_annotations_by_type(events, "error")
        assert any("INTERNAL" in a["error"]["code"] for a in error_anns)
        stream_errors = [e for e in events if e.get("type") == "error"]
        assert any("Generation failed" in e["error"] for e in stream_errors)


# ---------------------------------------------------------------------------
# SSE stream format validation
# ---------------------------------------------------------------------------


class TestStreamLineFormat:
    @pytest.mark.anyio
    async def test_sse_stream_is_non_empty_and_parseable(self, mock_engine) -> None:
        """SSE stream response is non-empty, all frames are valid JSON."""
        with patch("src.api._get_engine", return_value=mock_engine):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/api/chat",
                    json={"messages": [{"role": "user", "content": "Test"}]},
                )

        assert resp.status_code == 200
        body = resp.text
        assert len(body) > 0
        events = _parse_sse_events(body)  # raises on malformed JSON
        assert len(events) > 0

    @pytest.mark.anyio
    async def test_stream_starts_with_start_frame(self, mock_engine) -> None:
        """First SSE frame is the mandatory UI message stream start frame."""
        with patch("src.api._get_engine", return_value=mock_engine):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/api/chat",
                    json={"messages": [{"role": "user", "content": "End test"}]},
                )

        events = _parse_sse_events(resp.text)
        assert events[0]["type"] == "start"
        assert "messageId" in events[0]

    @pytest.mark.anyio
    async def test_stream_ends_with_finish_and_done(self, mock_engine) -> None:
        """SSE stream ends with a finish frame followed by the [DONE] sentinel."""
        with patch("src.api._get_engine", return_value=mock_engine):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/api/chat",
                    json={"messages": [{"role": "user", "content": "End test"}]},
                )

        assert resp.status_code == 200
        # Last parseable event must be finish; [DONE] sentinel follows
        events = _parse_sse_events(resp.text)
        assert events[-1]["type"] == "finish"
        assert resp.text.rstrip().endswith("[DONE]")

    @pytest.mark.anyio
    async def test_sse_answer_text_content(self, mock_engine) -> None:
        """Concatenated text-delta frames reproduce the full answer."""
        with patch("src.api._get_engine", return_value=mock_engine):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/api/chat",
                    json={"messages": [{"role": "user", "content": "End test"}]},
                )

        events = _parse_sse_events(resp.text)
        assert _get_sse_text(events) == "Answer to: End test"
