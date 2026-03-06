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
    image_count: Optional[int] = Field(
        default=None,
        description="Number of extracted images/figures for this source (Phase 4).",
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
# Analytics
# ---------------------------------------------------------------------------


class CorpusOverview(BaseModel):
    """High-level corpus statistics."""

    model_config = ConfigDict(extra="forbid")

    source_count: int = 0
    child_chunk_count: int = 0
    parent_chunk_count: int = 0
    estimated_tokens: int = 0
    avg_chunks_per_doc: float = 0.0
    source_ids: list[str] = Field(default_factory=list)


class TopicCluster(BaseModel):
    """A single TF-IDF topic cluster."""

    model_config = ConfigDict(extra="forbid")

    cluster_id: int
    label: str
    keywords: list[str]
    source_ids: list[str]
    size: int


class EntityFrequency(BaseModel):
    """Named entity with frequency count."""

    model_config = ConfigDict(extra="forbid")

    text: str
    type: str
    count: int


class TimelineBucket(BaseModel):
    """A temporal distribution bucket (decade)."""

    model_config = ConfigDict(extra="forbid")

    period_start: int
    period_end: int
    label: str
    count: int
    sources: list[str]


class RelationshipNode(BaseModel):
    """A single source node in the relationship graph."""

    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    size: int = 0
    dominant_topic: Optional[int] = None
    summary: Optional[str] = None


class RelationshipEdge(BaseModel):
    """A weighted edge between two source nodes."""

    model_config = ConfigDict(extra="forbid")

    source: str
    target: str
    types: list[str]
    weights: dict[str, float]
    combined_weight: float


class RelationshipGraph(BaseModel):
    """Nodes + edges for the source relationship force graph."""

    model_config = ConfigDict(extra="forbid")

    nodes: list[RelationshipNode] = Field(default_factory=list)
    edges: list[RelationshipEdge] = Field(default_factory=list)


class AnalyticsResponse(BaseModel):
    """Full corpus analytics response."""

    model_config = ConfigDict(extra="forbid")

    overview: CorpusOverview = Field(default_factory=CorpusOverview)
    topics: list[TopicCluster] = Field(default_factory=list)
    entities: list[EntityFrequency] = Field(default_factory=list)
    timeline: list[TimelineBucket] = Field(default_factory=list)
    relationships: RelationshipGraph = Field(default_factory=RelationshipGraph)
    ner_available: bool = False
    timeline_available: bool = True


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
    spacy_available: bool = False
    image_extraction_enabled: bool = True
    analytics_cache_status: Optional[str] = Field(
        default=None,
        description="Analytics cache status: 'fresh', 'stale', or 'empty'.",
    )
