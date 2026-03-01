"""Tests for AI SDK UI message stream (SSE) encoder.

Validates exact SSE frame format, JSON validity, double-newline termination,
and edge cases (empty strings, unicode, special characters).

Protocol format::

    data: {json_payload}\\n\\n
"""

from __future__ import annotations

import json

import pytest

from src.stream_protocol import (
    STREAM_HEADERS,
    annotation_error,
    annotation_intent,
    annotation_sources,
    annotation_status,
    encode_annotation,
    encode_data,
    encode_error,
    encode_finish_message,
    encode_finish_step,
    encode_text_delta,
    encode_text_end,
    encode_text_start,
    http_error_body,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_sse_frame(frame: str) -> dict:
    """Parse one SSE frame (``data: {...}\\n\\n``) into a Python dict."""
    frame = frame.strip()
    assert frame.startswith("data: "), f"Frame must start with 'data: ': {frame!r}"
    payload = frame[6:]  # strip "data: "
    return json.loads(payload)


def _parse_sse_body(body: str) -> list[dict]:
    """Split a stream body into individual SSE frames and parse each one.

    Skips the terminal ``data: [DONE]`` sentinel if present.
    """
    events: list[dict] = []
    for chunk in body.split("\n\n"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if chunk == "data: [DONE]":
            continue
        events.append(_parse_sse_frame(chunk))
    return events


def _encode_text_inline(token: str) -> str:
    """Test helper: emit text-start + text-delta + text-end as one call."""
    return encode_text_start() + encode_text_delta(token) + encode_text_end()



# ---------------------------------------------------------------------------
# encode_text_start / encode_text_delta / encode_text_end
# ---------------------------------------------------------------------------


class TestEncodeText:
    def test_simple_token_produces_three_frames(self) -> None:
        body = _encode_text_inline("Hello")
        events = _parse_sse_body(body)
        assert len(events) == 3
        assert events[0] == {"type": "text-start", "id": "text-0"}
        assert events[1] == {"type": "text-delta", "id": "text-0", "delta": "Hello"}
        assert events[2] == {"type": "text-end", "id": "text-0"}

    def test_empty_string(self) -> None:
        events = _parse_sse_body(_encode_text_inline(""))
        assert events[1]["delta"] == ""

    def test_unicode_characters(self) -> None:
        text = "Héllo wörld 你好 🌍"
        events = _parse_sse_body(_encode_text_inline(text))
        assert events[1]["delta"] == text

    def test_special_json_characters(self) -> None:
        """Quotes, backslashes, and newlines must survive JSON round-trip."""
        text = 'He said "hello"\nand\\walked away'
        events = _parse_sse_body(_encode_text_inline(text))
        assert events[1]["delta"] == text

    def test_single_space(self) -> None:
        events = _parse_sse_body(_encode_text_inline(" "))
        assert events[1]["delta"] == " "

    def test_frame_terminator(self) -> None:
        """Every SSE frame must end with \\n\\n."""
        body = _encode_text_inline("hi")
        for frame in body.split("\n\n"):
            if frame.strip():
                assert frame.startswith("data: ")


# ---------------------------------------------------------------------------
# encode_data
# ---------------------------------------------------------------------------


class TestEncodeData:
    def test_single_object_with_type(self) -> None:
        body = encode_data([{"type": "status", "status": "Loading"}])
        events = _parse_sse_body(body)
        assert len(events) == 1
        assert events[0]["type"] == "data-status"
        assert events[0]["data"] == {"type": "status", "status": "Loading"}

    def test_empty_array_produces_no_frames(self) -> None:
        body = encode_data([])
        events = _parse_sse_body(body)
        assert events == []

    def test_multiple_items_produce_multiple_frames(self) -> None:
        body = encode_data([
            {"type": "status", "status": "a"},
            {"type": "sources", "sourceIds": ["x"]},
        ])
        events = _parse_sse_body(body)
        assert len(events) == 2
        assert events[0]["type"] == "data-status"
        assert events[1]["type"] == "data-sources"

    def test_nested_objects_survive_round_trip(self) -> None:
        item = {"type": "custom", "nested": {"a": [1, 2, 3]}}
        events = _parse_sse_body(encode_data([item]))
        assert events[0]["data"] == item


# ---------------------------------------------------------------------------
# encode_error
# ---------------------------------------------------------------------------


class TestEncodeError:
    def test_simple_error(self) -> None:
        body = encode_error("Something went wrong")
        assert body == 'data: {"type": "error", "error": "Something went wrong"}\n\n'
        events = _parse_sse_body(body)
        assert events[0] == {"type": "error", "error": "Something went wrong"}

    def test_error_with_quotes(self) -> None:
        msg = 'File "test.pdf" not found'
        events = _parse_sse_body(encode_error(msg))
        assert events[0]["error"] == msg

    def test_empty_error(self) -> None:
        events = _parse_sse_body(encode_error(""))
        assert events[0]["error"] == ""


# ---------------------------------------------------------------------------
# encode_annotation
# ---------------------------------------------------------------------------


class TestEncodeAnnotation:
    def test_single_annotation_with_known_type(self) -> None:
        body = encode_annotation([{"type": "status", "status": "Loading..."}])
        events = _parse_sse_body(body)
        assert len(events) == 1
        assert events[0]["type"] == "data-status"
        assert events[0]["data"]["status"] == "Loading..."

    def test_empty_annotations_produce_no_frames(self) -> None:
        body = encode_annotation([])
        assert _parse_sse_body(body) == []

    def test_multiple_annotations_produce_multiple_frames(self) -> None:
        annotations = [
            {"type": "status", "status": "step1"},
            {"type": "sources", "sourceIds": ["a"]},
        ]
        events = _parse_sse_body(encode_annotation(annotations))
        assert len(events) == 2


# ---------------------------------------------------------------------------
# encode_finish_message
# ---------------------------------------------------------------------------


class TestEncodeFinishMessage:
    def test_default_finish_emits_type_finish(self) -> None:
        body = encode_finish_message()
        assert body == 'data: {"type": "finish"}\n\n'
        events = _parse_sse_body(body)
        assert events[0] == {"type": "finish"}

    def test_params_do_not_appear_on_wire(self) -> None:
        """finish_reason is accepted but output is always just {type: finish}."""
        body = encode_finish_message("length")
        events = _parse_sse_body(body)
        assert events[0] == {"type": "finish"}

    def test_error_reason_produces_same_frame(self) -> None:
        body = encode_finish_message("error")
        events = _parse_sse_body(body)
        assert events[0]["type"] == "finish"


# ---------------------------------------------------------------------------
# encode_finish_step
# ---------------------------------------------------------------------------


class TestEncodeFinishStep:
    def test_default_step(self) -> None:
        body = encode_finish_step()
        events = _parse_sse_body(body)
        assert len(events) == 1
        assert events[0]["type"] == "data-finish-step"
        data = events[0]["data"]
        assert data["finishReason"] == "stop"
        assert data["isContinued"] is False

    def test_continued_step(self) -> None:
        events = _parse_sse_body(encode_finish_step("stop", is_continued=True))
        assert events[0]["data"]["isContinued"] is True

    def test_error_reason(self) -> None:
        events = _parse_sse_body(encode_finish_step("error"))
        assert events[0]["data"]["finishReason"] == "error"


# ---------------------------------------------------------------------------
# High-level annotation helpers
# ---------------------------------------------------------------------------


class TestAnnotationHelpers:
    def test_status(self) -> None:
        body = annotation_status("Classifying intent...")
        assert body == (
            'data: {"type": "data-status", "data": {"type": "status",'
            ' "status": "Classifying intent..."}}\n\n'
        )
        events = _parse_sse_body(body)
        assert events[0]["type"] == "data-status"
        assert events[0]["data"]["status"] == "Classifying intent..."

    def test_sources(self) -> None:
        events = _parse_sse_body(annotation_sources(["doc_a", "doc_b"]))
        assert events[0]["type"] == "data-sources"
        assert events[0]["data"]["sourceIds"] == ["doc_a", "doc_b"]

    def test_sources_empty(self) -> None:
        events = _parse_sse_body(annotation_sources([]))
        assert events[0]["data"]["sourceIds"] == []

    def test_intent(self) -> None:
        events = _parse_sse_body(annotation_intent("analyze", 0.85, "heuristic"))
        assert events[0]["type"] == "data-intent"
        d = events[0]["data"]
        assert d["intent"] == "analyze"
        assert d["confidence"] == 0.85
        assert d["method"] == "heuristic"

    def test_intent_low_confidence(self) -> None:
        events = _parse_sse_body(annotation_intent("overview", 0.40, "fallback"))
        assert events[0]["data"]["confidence"] == 0.40

    def test_error_annotation(self) -> None:
        events = _parse_sse_body(annotation_error("INTERNAL", "Generation failed"))
        assert events[0]["type"] == "data-error"
        assert events[0]["data"]["error"]["code"] == "INTERNAL"
        assert events[0]["data"]["error"]["message"] == "Generation failed"

    def test_error_annotation_all_codes(self) -> None:
        """Verify all error codes from the contract can be encoded."""
        for error_code in [
            "SOURCE_NOT_FOUND",
            "INGEST_FAILED",
            "STREAM_CANCELLED",
            "INTERNAL",
        ]:
            events = _parse_sse_body(annotation_error(error_code, f"Test {error_code}"))
            assert events[0]["data"]["error"]["code"] == error_code


# ---------------------------------------------------------------------------
# HTTP error body
# ---------------------------------------------------------------------------


class TestHttpErrorBody:
    def test_structure(self) -> None:
        body = http_error_body("SOURCE_NOT_FOUND", "Document not found")
        assert body == {
            "error": {
                "code": "SOURCE_NOT_FOUND",
                "message": "Document not found",
            }
        }

    def test_serializable(self) -> None:
        body = http_error_body("INTERNAL", "Unexpected error")
        serialized = json.dumps(body)
        deserialized = json.loads(serialized)
        assert deserialized == body


# ---------------------------------------------------------------------------
# STREAM_HEADERS
# ---------------------------------------------------------------------------


class TestStreamHeaders:
    def test_content_type(self) -> None:
        assert STREAM_HEADERS["Content-Type"] == "text/event-stream; charset=utf-8"

    def test_ui_message_stream_header(self) -> None:
        assert STREAM_HEADERS["X-Vercel-AI-UI-Message-Stream"] == "v1"

    def test_no_cache(self) -> None:
        assert STREAM_HEADERS["Cache-Control"] == "no-cache"

    def test_no_buffering(self) -> None:
        assert STREAM_HEADERS["X-Accel-Buffering"] == "no"


# ---------------------------------------------------------------------------
# Integration: full stream sequence
# ---------------------------------------------------------------------------


class TestFullStreamSequence:
    """Simulate a complete chat stream and verify the full SSE byte sequence."""

    def test_happy_path_sequence(self) -> None:
        """A normal chat response: status → intent → text chunks → sources → finish."""
        frames: list[str] = []

        # 1. Status updates
        frames.append(annotation_status("Classifying intent..."))
        frames.append(annotation_intent("analyze", 0.85, "heuristic"))
        frames.append(annotation_status("Searching knowledge base..."))
        frames.append(annotation_status("Generating answer..."))

        # 2. Text tokens (using inline text helper)
        frames.append(_encode_text_inline("The "))
        frames.append(_encode_text_inline("theory "))
        frames.append(_encode_text_inline("of "))
        frames.append(_encode_text_inline("generative "))
        frames.append(_encode_text_inline("grammar..."))

        # 3. Sources
        frames.append(annotation_sources(["linguistics_doc"]))

        # 4. Finish
        frames.append(encode_finish_step("stop"))
        frames.append(encode_finish_message("stop"))

        # Verify: every frame is a parseable SSE event
        body = "".join(frames)
        events = _parse_sse_body(body)
        assert len(events) > 0

        # Text tokens produce 3 frames each; 5 tokens → 15 text frames
        text_delta_events = [e for e in events if e.get("type") == "text-delta"]
        assert len(text_delta_events) == 5

        # Stream ends with {"type": "finish"}
        assert events[-1] == {"type": "finish"}

    def test_error_mid_stream_sequence(self) -> None:
        """Error occurs after some text has been sent."""
        frames: list[str] = []

        frames.append(annotation_status("Generating answer..."))
        frames.append(_encode_text_inline("Partial "))
        frames.append(_encode_text_inline("response"))

        # Error occurs
        frames.append(annotation_error("INTERNAL", "Model crashed"))
        frames.append(encode_error("Model crashed"))

        # Finish with error reason
        frames.append(encode_finish_step("error"))
        frames.append(encode_finish_message("error"))

        body = "".join(frames)
        events = _parse_sse_body(body)

        # Error annotation appears before stream-level error
        error_ann = next(e for e in events if e.get("type") == "data-error")
        assert error_ann["data"]["error"]["code"] == "INTERNAL"

        # Stream-level error carries human-readable message
        stream_error = next(e for e in events if e.get("type") == "error")
        assert stream_error["error"] == "Model crashed"

        # Finish frame is last
        assert events[-1]["type"] == "finish"

    def test_cancellation_sequence(self) -> None:
        """Stream cancelled by client disconnect."""
        frames: list[str] = []

        frames.append(_encode_text_inline("Partial"))
        frames.append(annotation_error("STREAM_CANCELLED", "Client disconnected"))
        frames.append(encode_error("Client disconnected"))
        frames.append(encode_finish_step("error"))
        frames.append(encode_finish_message("error"))

        events = _parse_sse_body("".join(frames))
        assert any(e.get("type") == "error" for e in events)
        assert events[-1]["type"] == "finish"
