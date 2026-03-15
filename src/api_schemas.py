"""API request/response schemas for the FastAPI layer.

These Pydantic models define the HTTP boundary and are intentionally separate
from the domain models in ``models.py`` (Metadata, ParentChunk, ChildChunk).
Domain models represent storage-layer entities; API schemas represent
what clients send and receive.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Error envelope (shared across all error responses)
# ---------------------------------------------------------------------------


class ErrorDetail(BaseModel):
    """Structured error detail matching the error contract."""

    model_config = ConfigDict(extra="forbid")

    code: str = Field(
        ...,
        description="Machine-readable error code (UPPER_SNAKE_CASE)",
        examples=["LOCK_BUSY", "SOURCE_NOT_FOUND", "INGEST_FAILED", "STREAM_CANCELLED", "INTERNAL"],
    )
    message: str = Field(
        ...,
        description="Human-readable error message",
        examples=["Another query is already in progress"],
    )


class ErrorResponse(BaseModel):
    """Standard error response envelope.

    All HTTP error responses use this shape::

        {"error": {"code": "UPPER_SNAKE", "message": "human-readable"}}
    """

    model_config = ConfigDict(extra="forbid")

    error: ErrorDetail


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


class ChatMessagePart(BaseModel):
    """A single part of a message (AI SDK v6 format)."""

    model_config = ConfigDict(extra="allow")

    type: str = Field(..., description="Part type: text, file, reasoning, etc.")
    text: Optional[str] = Field(default=None, description="Text content (for type='text')")


class ChatMessage(BaseModel):
    """A single message in the chat history.

    Supports both AI SDK v3 format (``content`` string) and v6 format
    (``parts`` array).  The ``get_text()`` method extracts the text
    content regardless of format.
    """

    model_config = ConfigDict(extra="allow")

    role: str = Field(
        ...,
        description="Message role",
        examples=["user", "assistant", "system"],
    )
    content: Optional[str] = Field(
        default=None,
        description="Message text content (v3 format)",
    )
    parts: Optional[list[ChatMessagePart]] = Field(
        default=None,
        description="Message parts (v6 format)",
    )

    def get_text(self) -> str:
        """Extract text content from either v3 or v6 format."""
        # v3 format: plain content string
        if self.content is not None:
            return self.content

        # v6 format: extract text from parts array
        if self.parts:
            texts = [
                p.text for p in self.parts if p.type == "text" and p.text
            ]
            return " ".join(texts)

        return ""


class ChatRequest(BaseModel):
    """Request body for the ``POST /api/chat`` endpoint.

    Accepts both AI SDK v3 and v6 request formats:
    - v3: ``{ messages: [{role, content}] }``
    - v6: ``{ id, messages: [{role, parts: [{type, text}]}], trigger, messageId }``
    """

    model_config = ConfigDict(extra="allow")

    messages: list[ChatMessage] = Field(
        ...,
        min_length=1,
        description="Conversation history. Last message is the current user query.",
    )
    data: Optional[dict] = Field(
        default=None,
        description="Optional additional data from the frontend (source filters, settings).",
    )


class QueryRequest(BaseModel):
    """Request body for ``POST /api/query`` endpoint."""

    model_config = ConfigDict(extra="allow")

    query: str = Field(..., min_length=1, description="User query text")
    mode: Optional[str] = Field(default=None, description="Optional mode override")
    source_ids: Optional[list[str]] = Field(
        default=None,
        description="Optional selected source IDs (single source filter supported by backend)",
    )
    citations_enabled: bool = Field(
        default=True,
        description="Whether to include citation mapping events",
    )
    stream: bool = Field(default=False, description="Enable SSE streaming response")
    intent_override: Optional[str] = Field(
        default=None,
        description="Optional intent override; 'auto' or None uses automatic classification",
    )


class QueryResponse(BaseModel):
    """Non-streaming response for ``POST /api/query``."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    citations: list[dict[str, object]] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    metrics: dict[str, object] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Source management
# ---------------------------------------------------------------------------


class IngestRequest(BaseModel):
    """Request body for the ``POST /api/sources/ingest`` endpoint."""

    model_config = ConfigDict(extra="forbid")

    file_path: str = Field(
        ...,
        min_length=1,
        description="Absolute or relative path to the file to ingest (PDF or Markdown).",
    )
    source_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for this source document.",
    )
    summarize: bool = Field(
        default=True,
        description="Whether to generate a summary during ingest.",
    )
    page_offset: int = Field(
        default=1,
        ge=1,
        description="Starting page number for the first physical PDF page. No effect on Markdown files.",
    )


class IngestResponse(BaseModel):
    """Response body after successful ingestion."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    parents_count: int
    children_count: int
    summarized: bool


class SourceInfo(BaseModel):
    """Information about a single ingested source."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    summary: Optional[str] = Field(
        default=None,
        description="Source summary text, if available.",
    )
    source_path: Optional[str] = Field(
        default=None,
        description="Original file path used during ingest.",
    )
    snapshot_path: Optional[str] = Field(
        default=None,
        description="Path to cached text snapshot.",
    )
    source_size_bytes: Optional[int] = Field(
        default=None,
        description="Size of the original source file in bytes, if available.",
    )
    content_size_bytes: Optional[int] = Field(
        default=None,
        description="Best-available text/content size in bytes for UI display.",
    )
    page_offset: int = Field(
        default=1,
        description="Starting page number for the first physical PDF page.",
    )


class SourceListResponse(BaseModel):
    """Response body for ``GET /api/sources``."""

    model_config = ConfigDict(extra="forbid")

    sources: list[SourceInfo]


class SourceDeleteResponse(BaseModel):
    """Response body for ``DELETE /api/sources/{source_id}``."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    deleted: bool


class SourceContentResponse(BaseModel):
    """Response body for ``GET /api/sources/{source_id}/content``."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    content: str
    content_source: str = Field(
        ...,
        description="Where the content was resolved from: 'original', 'snapshot', or 'summary'.",
        examples=["original", "snapshot", "summary"],
    )
    format: str = Field(
        default="text",
        description="Content format derived from source file extension: 'pdf', 'markdown', or 'text'.",
        examples=["pdf", "markdown", "text"],
    )


class ChunkDetailResponse(BaseModel):
    """Response body for ``GET /api/sources/{source_id}/chunk/{chunk_id}``."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    chunk_id: str
    chunk_text: str
    parent_text: Optional[str] = None
    page_number: Optional[int] = None
    start_page: Optional[int] = None
    end_page: Optional[int] = None
    display_page: Optional[str] = None
    header_path: str = ""
    format: str = Field(
        default="text",
        description="Content format: 'pdf', 'markdown', or 'text'.",
    )
    source_path: Optional[str] = None


class ChunkBatchItem(BaseModel):
    """A single chunk detail element in a batch response."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    chunk_id: str
    chunk_text: str
    page_number: Optional[int] = None
    start_page: Optional[int] = None
    end_page: Optional[int] = None
    display_page: Optional[str] = None
    header_path: str = ""
    format: str = Field(
        default="text",
        description="Content format: 'pdf', 'markdown', or 'text'.",
    )
    source_path: Optional[str] = None


class ChunkBatchResponse(BaseModel):
    """Response body for ``GET /api/sources/{source_id}/chunks?ids=...``."""

    model_config = ConfigDict(extra="forbid")

    chunks: list[ChunkBatchItem]


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Response body for ``GET /api/health``."""

    model_config = ConfigDict(extra="forbid")

    status: str = "ok"
    engine_loaded: bool = False
    system_ram_gb: Optional[float] = Field(
        default=None,
        description="Detected system RAM in GB, used by frontend for capability gating.",
    )
    phoenix_configured: bool = Field(
        default=False,
        description="Whether Phoenix tracing is configured (via flags or env vars).",
    )
    phoenix_active: bool = Field(
        default=False,
        description="Whether Phoenix exporter/tracer is currently active.",
    )
    phoenix_project_name: Optional[str] = Field(
        default=None,
        description="Phoenix project name currently configured.",
    )
    phoenix_endpoint: Optional[str] = Field(
        default=None,
        description="Phoenix collector endpoint currently configured.",
    )
    phoenix_error: Optional[str] = Field(
        default=None,
        description="Last Phoenix initialization error, if any.",
    )
