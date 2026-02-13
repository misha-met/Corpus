"""Structured event types for streaming RAG query execution.

These events are yielded by ``RagEngine.query_events()`` and consumed by
the API layer to produce AI SDK Data Stream Protocol lines.  They are
simple dataclasses — no framework dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class StatusEvent:
    """Pipeline status update (e.g. "Classifying intent...")."""

    status: str


@dataclass(frozen=True)
class IntentEvent:
    """Intent classification result."""

    intent: str  # e.g. "analyze", "summarize"
    confidence: float
    method: str  # e.g. "heuristic", "llm-fallback"


@dataclass(frozen=True)
class SourcesEvent:
    """Source IDs matched by retrieval."""

    source_ids: list[str]


@dataclass(frozen=True)
class TextTokenEvent:
    """A single text token from LLM generation."""

    token: str


@dataclass(frozen=True)
class ErrorEvent:
    """An error that occurred during pipeline execution."""

    code: str  # e.g. "INTERNAL", "STREAM_CANCELLED"
    message: str


@dataclass(frozen=True)
class FinishEvent:
    """Pipeline execution complete."""

    finish_reason: str = "stop"  # "stop" or "error"
    prompt_tokens: int = 0
    completion_tokens: int = 0


# Union type for type checking
QueryEvent = StatusEvent | IntentEvent | SourcesEvent | TextTokenEvent | ErrorEvent | FinishEvent
