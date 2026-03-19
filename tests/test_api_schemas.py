"""Tests for API schema validation.

Validates Pydantic models enforce constraints correctly: required fields,
extra fields, type coercion, and serialization round-trips.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from src.api_schemas import (
    ChatMessage,
    ChatMessagePart,
    ChatRequest,
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    NERDiagnosticsResponse,
    PeopleListResponse,
    PeopleMergeRequest,
    PeopleMergeResponse,
    PersonMention,
    PersonMentionsResponse,
    PersonSummary,
    SourceContentResponse,
    SourceDeleteResponse,
    SourceInfo,
    SourceListResponse,
)


# ---------------------------------------------------------------------------
# ErrorResponse
# ---------------------------------------------------------------------------


class TestErrorResponse:
    def test_valid_error(self) -> None:
        err = ErrorResponse(error=ErrorDetail(code="LOCK_BUSY", message="Busy"))
        assert err.error.code == "LOCK_BUSY"
        assert err.error.message == "Busy"

    def test_serialization_matches_contract(self) -> None:
        err = ErrorResponse(
            error=ErrorDetail(code="INTERNAL", message="Something broke")
        )
        data = json.loads(err.model_dump_json())
        assert data == {"error": {"code": "INTERNAL", "message": "Something broke"}}

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ErrorDetail(code="X", message="Y", extra_field="Z")

    def test_missing_code_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ErrorDetail(message="No code")  # type: ignore[call-arg]

    def test_missing_message_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ErrorDetail(code="NO_MSG")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# ChatRequest
# ---------------------------------------------------------------------------


class TestChatRequest:
    def test_valid_single_message_v3(self) -> None:
        """v3 format: messages have role + content."""
        req = ChatRequest(messages=[ChatMessage(role="user", content="Hello")])
        assert len(req.messages) == 1
        assert req.messages[0].role == "user"
        assert req.messages[0].get_text() == "Hello"
        assert req.data is None

    def test_valid_single_message_v6(self) -> None:
        """v6 format: messages have role + parts array."""
        req = ChatRequest(
            messages=[
                ChatMessage(
                    role="user",
                    parts=[ChatMessagePart(type="text", text="Hello from v6")],
                )
            ]
        )
        assert req.messages[0].get_text() == "Hello from v6"

    def test_v6_full_request_body(self) -> None:
        """Simulate the exact body AI SDK v6 DefaultChatTransport sends."""
        raw = {
            "id": "chat-abc123",
            "messages": [
                {
                    "id": "msg-1",
                    "role": "user",
                    "parts": [{"type": "text", "text": "What is this?"}],
                }
            ],
            "trigger": "submit-message",
            "messageId": "msg-1",
        }
        req = ChatRequest(**raw)
        assert len(req.messages) == 1
        assert req.messages[0].get_text() == "What is this?"

    def test_v6_multi_part_message(self) -> None:
        """v6 message with multiple text parts."""
        msg = ChatMessage(
            role="user",
            parts=[
                ChatMessagePart(type="text", text="Hello"),
                ChatMessagePart(type="text", text="world"),
            ],
        )
        assert msg.get_text() == "Hello world"

    def test_v6_non_text_parts_ignored(self) -> None:
        """Non-text parts (file, reasoning) are ignored by get_text()."""
        msg = ChatMessage(
            role="user",
            parts=[
                ChatMessagePart(type="text", text="Question here"),
                ChatMessagePart(type="file"),  # no text
            ],
        )
        assert msg.get_text() == "Question here"

    def test_get_text_prefers_content_over_parts(self) -> None:
        """If both content and parts exist, content wins (v3 compat)."""
        msg = ChatMessage(
            role="user",
            content="from content",
            parts=[ChatMessagePart(type="text", text="from parts")],
        )
        assert msg.get_text() == "from content"

    def test_get_text_empty_message(self) -> None:
        """Message with no content and no parts returns empty string."""
        msg = ChatMessage(role="user")
        assert msg.get_text() == ""

    def test_valid_with_history(self) -> None:
        req = ChatRequest(
            messages=[
                ChatMessage(role="user", content="First"),
                ChatMessage(role="assistant", content="Response"),
                ChatMessage(role="user", content="Follow-up"),
            ]
        )
        assert len(req.messages) == 3

    def test_valid_with_data(self) -> None:
        req = ChatRequest(
            messages=[ChatMessage(role="user", content="Hello")],
            data={"source_id": "doc_a", "citations_enabled": True},
        )
        assert req.data["source_id"] == "doc_a"

    def test_empty_messages_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ChatRequest(messages=[])

    def test_ai_sdk_extra_fields_allowed(self) -> None:
        """useChat may send extra fields like id, createdAt; these should not fail."""
        msg = ChatMessage(role="user", content="Hi", id="msg_123", createdAt="2024-01-01")
        assert msg.role == "user"

    def test_chat_request_extra_fields_allowed(self) -> None:
        """useChat sends extra top-level fields; ChatRequest allows them."""
        req = ChatRequest(
            messages=[ChatMessage(role="user", content="Hi")],
            model="gpt-4",  # extra field from AI SDK
        )
        assert req.messages[0].get_text() == "Hi"


# ---------------------------------------------------------------------------
# IngestRequest
# ---------------------------------------------------------------------------


class TestIngestRequest:
    def test_valid_ingest(self) -> None:
        req = IngestRequest(file_path="/docs/paper.pdf", source_id="paper_1")
        assert req.file_path == "/docs/paper.pdf"
        assert req.source_id == "paper_1"
        assert req.summarize is True  # default
        assert req.peopletag is False

    def test_no_summarize(self) -> None:
        req = IngestRequest(
            file_path="doc.md", source_id="doc", summarize=False
        )
        assert req.summarize is False

    def test_peopletag_enabled(self) -> None:
        req = IngestRequest(
            file_path="doc.md",
            source_id="doc",
            peopletag=True,
        )
        assert req.peopletag is True

    def test_citation_reference_optional(self) -> None:
        req = IngestRequest(
            file_path="doc.md",
            source_id="doc",
            citation_reference="Smith (2024)",
        )
        assert req.citation_reference == "Smith (2024)"

    def test_empty_file_path_rejected(self) -> None:
        with pytest.raises(ValidationError):
            IngestRequest(file_path="", source_id="doc")

    def test_empty_source_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            IngestRequest(file_path="doc.pdf", source_id="")

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            IngestRequest(
                file_path="doc.pdf", source_id="doc", unknown_field="val"
            )


# ---------------------------------------------------------------------------
# IngestResponse
# ---------------------------------------------------------------------------


class TestIngestResponse:
    def test_valid_response(self) -> None:
        resp = IngestResponse(
            source_id="doc", parents_count=10, children_count=20, summarized=True
        )
        data = resp.model_dump()
        assert data["parents_count"] == 10
        assert data["children_count"] == 20
        assert data["geotag_ner"] is None
        assert data["peopletag_ner"] is None

    def test_response_with_ner_diagnostics(self) -> None:
        resp = IngestResponse(
            source_id="doc",
            parents_count=2,
            children_count=4,
            summarized=False,
            geotag_ner=NERDiagnosticsResponse(
                ner_available=False,
                method="regex_fallback",
                warning="GLiNER unavailable",
            ),
            peopletag_ner=NERDiagnosticsResponse(
                ner_available=False,
                method="empty",
            ),
        )
        data = resp.model_dump()
        assert data["geotag_ner"]["method"] == "regex_fallback"
        assert data["peopletag_ner"]["method"] == "empty"


# ---------------------------------------------------------------------------
# SourceInfo / SourceListResponse
# ---------------------------------------------------------------------------


class TestSourceListResponse:
    def test_empty_list(self) -> None:
        resp = SourceListResponse(sources=[])
        assert resp.sources == []

    def test_with_sources(self) -> None:
        resp = SourceListResponse(
            sources=[
                SourceInfo(source_id="doc_a", summary="A document about X"),
                SourceInfo(source_id="doc_b"),
            ]
        )
        assert len(resp.sources) == 2
        assert resp.sources[0].summary == "A document about X"
        assert resp.sources[1].summary is None

    def test_source_with_paths(self) -> None:
        info = SourceInfo(
            source_id="doc",
            source_path="/docs/paper.pdf",
            snapshot_path="data/source_cache/doc.txt",
        )
        assert info.source_path == "/docs/paper.pdf"

    def test_source_with_citation_reference(self) -> None:
        info = SourceInfo(source_id="doc", citation_reference="Doe (2023)")
        assert info.citation_reference == "Doe (2023)"


# ---------------------------------------------------------------------------
# SourceDeleteResponse
# ---------------------------------------------------------------------------


class TestSourceDeleteResponse:
    def test_deleted(self) -> None:
        resp = SourceDeleteResponse(source_id="doc", deleted=True)
        assert resp.deleted is True

    def test_not_found(self) -> None:
        resp = SourceDeleteResponse(source_id="doc", deleted=False)
        assert resp.deleted is False


# ---------------------------------------------------------------------------
# SourceContentResponse
# ---------------------------------------------------------------------------


class TestSourceContentResponse:
    def test_valid_content(self) -> None:
        resp = SourceContentResponse(
            source_id="doc",
            content="Full text of the document...",
            content_source="original",
        )
        assert resp.content_source == "original"

    def test_snapshot_source(self) -> None:
        resp = SourceContentResponse(
            source_id="doc",
            content="Cached text...",
            content_source="snapshot",
        )
        assert resp.content_source == "snapshot"


# ---------------------------------------------------------------------------
# HealthResponse
# ---------------------------------------------------------------------------


class TestHealthResponse:
    def test_defaults(self) -> None:
        resp = HealthResponse()
        assert resp.status == "ok"
        assert resp.engine_loaded is False

    def test_loaded(self) -> None:
        resp = HealthResponse(status="ok", engine_loaded=True)
        assert resp.engine_loaded is True

    def test_serialization(self) -> None:
        resp = HealthResponse(engine_loaded=True)
        data = json.loads(resp.model_dump_json())
        assert data["status"] == "ok"
        assert data["engine_loaded"] is True
        assert "system_ram_gb" in data  # optional field, may be None

    def test_fts_status_fields(self) -> None:
        resp = HealthResponse(
            engine_loaded=True,
            fts_policy="immediate",
            fts_dirty=False,
            fts_pending_rows=0,
        )
        data = json.loads(resp.model_dump_json())
        assert data["fts_policy"] == "immediate"
        assert data["fts_dirty"] is False
        assert data["fts_pending_rows"] == 0


# ---------------------------------------------------------------------------
# People schemas
# ---------------------------------------------------------------------------


class TestPeopleSchemas:
    def test_person_mention_model(self) -> None:
        mention = PersonMention(
            id="pm-1",
            source_id="doc_a",
            chunk_id="chunk-1",
            raw_name="Noam Chomsky",
            canonical_name="Noam Chomsky",
            confidence=0.94,
            method="exact",
            role_hint="author",
            context_snippet="...written by Noam Chomsky...",
        )
        assert mention.canonical_name == "Noam Chomsky"

    def test_people_list_response(self) -> None:
        resp = PeopleListResponse(
            count=1,
            people=[
                PersonSummary(
                    canonical_name="Noam Chomsky",
                    mention_count=3,
                    source_count=2,
                    source_ids=["doc_a", "doc_b"],
                    variants=["Noam Chomsky", "Chomsky"],
                    roles=["author"],
                    avg_confidence=0.91,
                )
            ],
        )
        assert resp.count == 1
        assert resp.people[0].mention_count == 3

    def test_person_mentions_response(self) -> None:
        resp = PersonMentionsResponse(
            canonical_name="Noam Chomsky",
            count=1,
            mentions=[
                PersonMention(
                    id="pm-1",
                    source_id="doc_a",
                    chunk_id="chunk-1",
                    raw_name="Noam Chomsky",
                    canonical_name="Noam Chomsky",
                    confidence=0.94,
                    method="exact",
                    role_hint="author",
                    context_snippet="context",
                )
            ],
        )
        assert resp.canonical_name == "Noam Chomsky"
        assert resp.count == 1

    def test_people_merge_request(self) -> None:
        req = PeopleMergeRequest(
            source_canonical_name="Collins",
            target_canonical_name="Michael Collins",
        )
        assert req.source_canonical_name == "Collins"
        assert req.target_canonical_name == "Michael Collins"

    def test_people_merge_response(self) -> None:
        resp = PeopleMergeResponse(
            source_canonical_name="Collins",
            target_canonical_name="Michael Collins",
            merged_count=4,
        )
        assert resp.merged_count == 4
