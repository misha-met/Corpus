from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from .models import ChildChunk, Metadata, ParentChunk

HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$")

PARENT_MIN_TOKENS = 1000
PARENT_MAX_TOKENS = 1500
PARENT_TARGET_TOKENS = 1200
PARENT_OVERLAP_TOKENS = 150

CHILD_MIN_TOKENS = 200
CHILD_MAX_TOKENS = 300
CHILD_TARGET_TOKENS = 250
CHILD_OVERLAP_TOKENS = 50


@dataclass(frozen=True)
class _Section:
    header_path: str
    text: str


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def _detokenize(tokens: Iterable[str]) -> str:
    return " ".join(tokens).strip()


def _split_tokens(tokens: list[str], chunk_size: int, overlap: int) -> list[list[str]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size.")

    if not tokens:
        return []

    chunks: list[list[str]] = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(tokens[start:end])
        if end == len(tokens):
            break
        start = end - overlap
    return chunks


def _parse_markdown_sections(text: str) -> list[_Section]:
    lines = text.splitlines()
    header_stack: list[tuple[int, str]] = []
    sections: list[_Section] = []

    current_lines: list[str] = []
    current_header_path = "Document"

    for line in lines:
        match = HEADER_RE.match(line)
        if match:
            if current_lines:
                section_text = "\n".join(current_lines).strip()
                if section_text:
                    sections.append(_Section(current_header_path, section_text))
                current_lines = []

            level = len(match.group(1))
            title = match.group(2).strip()
            header_stack = [(lvl, t) for (lvl, t) in header_stack if lvl < level]
            header_stack.append((level, title))
            current_header_path = " > ".join(t for _, t in header_stack) or "Document"
            current_lines.append(line)
        else:
            current_lines.append(line)

    if current_lines:
        section_text = "\n".join(current_lines).strip()
        if section_text:
            sections.append(_Section(current_header_path, section_text))

    return sections


def _split_parent_chunks(
    section: _Section,
    *,
    source_id: str,
    page_number: Optional[int],
) -> list[ParentChunk]:
    tokens = _tokenize(section.text)
    if not tokens:
        return []

    if len(tokens) <= PARENT_MAX_TOKENS:
        token_chunks = [tokens]
    else:
        token_chunks = _split_tokens(tokens, PARENT_TARGET_TOKENS, PARENT_OVERLAP_TOKENS)
        if len(token_chunks) > 1 and len(token_chunks[-1]) < PARENT_MIN_TOKENS:
            token_chunks[-2].extend(token_chunks[-1])
            token_chunks.pop()

    parents: list[ParentChunk] = []
    for chunk_tokens in token_chunks:
        text = _detokenize(chunk_tokens)
        metadata = Metadata(
            source_id=source_id,
            page_number=page_number,
            header_path=section.header_path,
            parent_id=None,
        )
        parents.append(ParentChunk(text=text, metadata=metadata))

    return parents


def _split_child_chunks(parent: ParentChunk) -> list[ChildChunk]:
    tokens = _tokenize(parent.text)
    if not tokens:
        return []

    token_chunks = _split_tokens(tokens, CHILD_TARGET_TOKENS, CHILD_OVERLAP_TOKENS)
    child_chunks: list[ChildChunk] = []

    for chunk_tokens in token_chunks:
        if len(chunk_tokens) < CHILD_MIN_TOKENS:
            continue
        if len(chunk_tokens) > CHILD_MAX_TOKENS:
            chunk_tokens = chunk_tokens[:CHILD_MAX_TOKENS]
        text = _detokenize(chunk_tokens)
        metadata = Metadata(
            source_id=parent.metadata.source_id,
            page_number=parent.metadata.page_number,
            header_path=parent.metadata.header_path,
            parent_id=parent.id,
        )
        child_chunks.append(ChildChunk(text=text, metadata=metadata))

    return child_chunks


def ingest_markdown(
    file_path: str | Path,
    *,
    source_id: str,
    page_number: Optional[int] = None,
) -> tuple[list[ParentChunk], list[ChildChunk]]:
    path = Path(file_path)
    if not source_id or not source_id.strip():
        raise ValueError("source_id must be a non-empty string.")
    if not path.exists():
        raise FileNotFoundError(f"Markdown file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("Markdown file is empty.")

    sections = _parse_markdown_sections(text)
    if not sections:
        raise ValueError("No parsable content found in markdown file.")

    parents: list[ParentChunk] = []
    children: list[ChildChunk] = []

    for section in sections:
        for parent in _split_parent_chunks(
            section,
            source_id=source_id.strip(),
            page_number=page_number,
        ):
            parents.append(parent)
            children.extend(_split_child_chunks(parent))

    if not parents:
        raise ValueError("No parent chunks produced from markdown content.")
    if not children:
        raise ValueError("No child chunks produced from markdown content.")

    return parents, children
