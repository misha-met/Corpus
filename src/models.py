from __future__ import annotations

import uuid
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class Metadata(BaseModel):
    """Metadata for document chunks.
    
    Attributes:
        source_id: Unique identifier for the source document
        page_number: Physical page number (1-indexed, from PDF page order)
        page_label: Logical page label from PDF (e.g., 'iii', 'xii', '1', '2')
        display_page: Human-readable page for citations (page_label or str(page_number))
        header_path: Hierarchical header path (e.g., "Chapter 1 > Section 1.1")
        parent_id: ID of parent chunk (for child chunks only)
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(..., min_length=1)
    page_number: Optional[int] = Field(default=None, ge=1)
    page_label: Optional[str] = Field(default=None, description="Logical page label from PDF")
    display_page: Optional[str] = Field(default=None, description="Human-readable page for citations")
    header_path: str = Field(..., min_length=1)
    parent_id: Optional[str] = None


class ParentChunk(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str = Field(..., min_length=1)
    metadata: Metadata


class ChildChunk(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str = Field(..., min_length=1)
    metadata: Metadata
