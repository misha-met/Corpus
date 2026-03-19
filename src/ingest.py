"""Document ingestion pipeline: parsing → chunking → storage.

Supports PDF and Markdown source documents.

Architecture
~~~~~~~~~~~~
- PDF extraction cascades through four strategies in order: PyMuPDF →
  PyPDF → pdfminer → Tesseract OCR.  The first strategy that returns
  non-empty text wins; later strategies are only tried if earlier ones fail
  or return empty output.
- Text is chunked into overlapping parent chunks (~1 200 tokens with 150-
  token overlap) and nested child chunks (~250 tokens with 2-sentence
  overlap).  This parent-document retrieval layout means the reranker scores
  focused child passages while the LLM sees the wider parent context.
- ``ingest_file_to_storage()`` is the main entry point; it returns lists of
  ``ParentChunk`` and ``ChildChunk`` objects that the caller (``RagEngine``)
  writes to ``StorageEngine``.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, TYPE_CHECKING

from .models import ChildChunk, Metadata, ParentChunk
from .generation import build_ingest_summary_messages
from .generator import MlxGenerator
from .phoenix_tracing import (
    SPAN_KIND_CHAIN,
    SPAN_KIND_EMBEDDING,
    SPAN_KIND_LLM,
    mark_span_error,
    set_span_attributes,
    start_span,
)
from .storage import StorageEngine

if TYPE_CHECKING:
    from .ner import NERDiagnostics

logger = logging.getLogger(__name__)

_SUMMARY_CONTEXT_CHAR_LIMIT = 12_000


def _sample_context(text: str, limit: int) -> str:
    """Return a representative sample of *text* up to *limit* characters.

    If the text is within the limit it is returned unchanged.  Otherwise the
    head, middle, and tail thirds are joined with a '[...]' separator so the
    summary model sees the beginning, a mid-document slice, and the end.
    """
    if len(text) <= limit:
        return text
    third = limit // 3
    head = text[:third]
    mid_center = len(text) // 2
    middle = text[mid_center - third // 2 : mid_center + third // 2]
    tail = text[-third:]
    return "\n\n[...]\n\n".join([head, middle, tail])

HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$")

PARENT_MIN_TOKENS = 1000
PARENT_MAX_TOKENS = 1500
PARENT_TARGET_TOKENS = 1200
PARENT_OVERLAP_TOKENS = 150

CHILD_MIN_TOKENS = 120
CHILD_MAX_TOKENS = 250
CHILD_TARGET_TOKENS = 200
CHILD_OVERLAP_SENTENCES = 2

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
CLAUSE_BOUNDARY_RE = re.compile(r";\s+|,\s+(?:and|but|which|or)\s+", re.IGNORECASE)
PAGE_MARKER_RE = re.compile(r"\[Page\s+(\d+)\]", re.IGNORECASE)


@dataclass(frozen=True)
class _Section:
    header_path: str
    text: str


def _format_page_marker(page_number: int) -> str:
    page = int(page_number)
    if page < 1:
        page = 1
    return f"[Page {page}]"


def _parse_page_markers(text: str) -> list[int]:
    if not text:
        return []

    pages: list[int] = []
    for match in PAGE_MARKER_RE.finditer(text):
        try:
            page = int(match.group(1))
        except (TypeError, ValueError):
            continue
        if page >= 1:
            pages.append(page)
    return pages


def _marker_page_range(text: str) -> tuple[Optional[int], Optional[int]]:
    pages = _parse_page_markers(text)
    if not pages:
        return None, None
    return pages[0], pages[-1]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def _detokenize(tokens: Iterable[str]) -> str:
    return " ".join(tokens).strip()


def _token_count(text: str) -> int:
    return len(_tokenize(text))


def clean_ocr_artifacts(text: str) -> str:
    if not text:
        return text

    # Merge hyphenated line breaks like "lin-\n guists" -> "linguists".
    text = re.sub(r"([A-Za-z])-[\t\r\n ]+([A-Za-z])", r"\1\2", text)
    return text


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


def _split_sentences(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    return [sentence.strip() for sentence in SENTENCE_SPLIT_RE.split(stripped) if sentence.strip()]


def _split_long_sentence_on_clause(sentence: str, target_tokens: int) -> list[str]:
    """Recursively split a single long sentence into smaller units.
    
    Tries to split at midpoint clause boundaries (comma, semicolon, conjunctions).
    If no clause boundary is found and the sentence is still too long, falls
    back to a forced split by word count to ensure GLiNER token limits are met.
    """
    total_tokens = _token_count(sentence)
    if total_tokens <= target_tokens:
        return [sentence]

    boundary_matches = list(CLAUSE_BOUNDARY_RE.finditer(sentence))
    split_index: int | None = None

    if boundary_matches:
        midpoint = len(sentence) / 2
        best = min(boundary_matches, key=lambda match: abs(match.start() - midpoint))
        split_index = best.start() + 1 if sentence[best.start()] in {",", ";"} else best.start()
    elif total_tokens > int(target_tokens * 1.2):
        # Fallback: force split at word midpoint if no elegant boundary exists
        words = _tokenize(sentence)
        mid_word = len(words) // 2
        left_text = _detokenize(words[:mid_word])
        split_index = len(left_text)

    if split_index is None:
        return [sentence]

    left = sentence[:split_index].strip()
    right = sentence[split_index:].strip()
    if not left or not right:
        return [sentence]

    # Recursively split both halves if they remain too long
    return _split_long_sentence_on_clause(left, target_tokens) + _split_long_sentence_on_clause(right, target_tokens)


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
    page_label: Optional[str] = None,
    display_page: Optional[str] = None,
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
        start_page, end_page = _marker_page_range(text)
        resolved_page_number = start_page if start_page is not None else page_number
        resolved_display_page = display_page if display_page is not None else (
            str(start_page) if start_page is not None else None
        )
        metadata = Metadata(
            source_id=source_id,
            page_number=resolved_page_number,
            start_page=start_page,
            end_page=end_page,
            page_label=page_label,
            display_page=resolved_display_page,
            header_path=section.header_path,
            parent_id=None,
        )
        parents.append(ParentChunk(text=text, metadata=metadata))

    return parents


def _split_child_chunks(parent: ParentChunk) -> list[ChildChunk]:
    sentences = _split_sentences(parent.text)
    if not sentences:
        return []

    units: list[str] = []
    for sentence in sentences:
        units.extend(_split_long_sentence_on_clause(sentence, CHILD_TARGET_TOKENS))

    sentence_chunks: list[list[str]] = []
    current_chunk: list[str] = []
    current_tokens = 0

    for unit in units:
        unit_tokens = _token_count(unit)

        if current_chunk and (current_tokens + unit_tokens) > CHILD_TARGET_TOKENS:
            sentence_chunks.append(current_chunk)
            overlap_sentences = current_chunk[-CHILD_OVERLAP_SENTENCES:] if CHILD_OVERLAP_SENTENCES > 0 else []
            current_chunk = list(overlap_sentences)
            current_tokens = sum(_token_count(sentence) for sentence in current_chunk)

        current_chunk.append(unit)
        current_tokens += unit_tokens

    if current_chunk:
        sentence_chunks.append(current_chunk)

    child_chunks: list[ChildChunk] = []
    _merge_count = 0
    _sole_child_count = 0

    def _build_child_metadata(child_text: str) -> Metadata:
        inferred_start, inferred_end = _marker_page_range(child_text)
        start_page = inferred_start if inferred_start is not None else parent.metadata.start_page
        end_page = inferred_end if inferred_end is not None else parent.metadata.end_page

        if inferred_start is not None:
            resolved_page_number = inferred_start
            resolved_display_page = str(inferred_start)
        else:
            resolved_page_number = parent.metadata.page_number
            resolved_display_page = parent.metadata.display_page

        return Metadata(
            source_id=parent.metadata.source_id,
            page_number=resolved_page_number,
            start_page=start_page,
            end_page=end_page,
            page_label=parent.metadata.page_label,
            display_page=resolved_display_page,
            header_path=parent.metadata.header_path,
            parent_id=parent.id,
        )

    for chunk_sentences in sentence_chunks:
        text = " ".join(chunk_sentences).strip()
        token_count = _token_count(text)
        if token_count < CHILD_MIN_TOKENS:
            # Instead of dropping, merge into previous child or create sole child
            if child_chunks:
                # Merge into previous child's text
                prev = child_chunks[-1]
                merged_text = prev.text + " " + text
                merged_metadata = _build_child_metadata(merged_text)
                # Re-create with same id and metadata (frozen model)
                child_chunks[-1] = ChildChunk.model_construct(
                    id=prev.id, text=merged_text, metadata=merged_metadata,
                )
                _merge_count += 1
            else:
                # No previous child — create a sole child (short but searchable)
                metadata = _build_child_metadata(text)
                child_chunks.append(ChildChunk(text=text, metadata=metadata))
                _sole_child_count += 1
            continue

        metadata = _build_child_metadata(text)
        child_chunks.append(ChildChunk(text=text, metadata=metadata))

    if _merge_count or _sole_child_count:
        logger.info(
            "Child merge: %d short segments merged, %d sole-child chunks created",
            _merge_count, _sole_child_count,
        )

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

    # For markdown, display_page is simply str(page_number) if provided
    display_page = str(page_number) if page_number is not None else None

    parents: list[ParentChunk] = []
    children: list[ChildChunk] = []

    for section in sections:
        for parent in _split_parent_chunks(
            section,
            source_id=source_id.strip(),
            page_number=page_number,
            page_label=None,
            display_page=display_page,
        ):
            parents.append(parent)
            children.extend(_split_child_chunks(parent))

    if not parents:
        raise ValueError("No parent chunks produced from markdown content.")
    if not children:
        raise ValueError("No child chunks produced from markdown content.")

    return parents, children


@dataclass
class _PageData:
    """Extracted text and metadata for a single PDF page (or whole-doc fallback)."""
    text: str
    page_number: Optional[int]
    page_label: Optional[str]
    display_page: Optional[str]


@dataclass(frozen=True)
class IngestNERDiagnostics:
    """NER diagnostics returned by ingest-time entity extraction helpers."""

    geotag_ner: Optional["NERDiagnostics"] = None
    peopletag_ner: Optional["NERDiagnostics"] = None


def _chunk_pages(
    pages: list[_PageData],
    source_id: str,
) -> tuple[list[ParentChunk], list[ChildChunk]]:
    """Convert extracted page data into parent/child chunk pairs."""
    parents: list[ParentChunk] = []
    children: list[ChildChunk] = []
    sid = source_id.strip()
    for page in pages:
        section = _Section(header_path="Document", text=page.text)
        for parent in _split_parent_chunks(
            section,
            source_id=sid,
            page_number=page.page_number,
            page_label=page.page_label,
            display_page=page.display_page,
        ):
            parents.append(parent)
            children.extend(_split_child_chunks(parent))
    return parents, children


def _extract_pymupdf(path: Path, *, page_offset: int = 1, **_kw) -> list[_PageData]:
    """Strategy 1: PyMuPDF per-page extraction."""
    try:
        import fitz
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyMuPDF is required for primary PDF extraction.") from exc
    doc = fitz.open(str(path))
    pages: list[_PageData] = []
    for index in range(doc.page_count):
        page = doc.load_page(index)
        page_text = clean_ocr_artifacts((page.get_text("text") or "").strip())
        if not page_text:
            continue
        page_label: Optional[str] = None
        try:
            if hasattr(page, "get_label"):
                page_label = page.get_label()
        except Exception:
            pass
        page_number = index + page_offset
        display_page = str(page_number) if page_offset != 1 else (page_label if page_label else str(page_number))
        marked_text = f"{_format_page_marker(page_number)}\n{page_text}"
        pages.append(_PageData(marked_text, page_number, page_label, display_page))
    return pages


def _extract_pypdf(reader, *, page_offset: int = 1, **_kw) -> list[_PageData]:
    """Strategy 2: pypdf per-page fallback."""
    pages: list[_PageData] = []
    for index, page in enumerate(reader.pages, start=1):
        page_text = clean_ocr_artifacts((page.extract_text() or "").strip())
        if not page_text:
            continue
        page_label: Optional[str] = None
        try:
            if hasattr(page, "get_label"):
                page_label = page.get_label()
        except Exception:
            pass
        page_number = (index - 1) + page_offset
        display_page = str(page_number) if page_offset != 1 else (page_label if page_label else str(index))
        marked_text = f"{_format_page_marker(page_number)}\n{page_text}"
        pages.append(_PageData(marked_text, page_number, page_label, display_page))
    return pages


def _extract_pdfminer(path: Path, *, reader, page_offset: int = 1, **_kw) -> list[_PageData]:
    """Strategy 3: pdfminer per-page fallback."""
    try:
        from pdfminer.high_level import extract_text
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pdfminer.six is required for fallback PDF extraction.") from exc

    pages: list[_PageData] = []
    page_count = len(reader.pages)
    for index in range(page_count):
        try:
            raw_text = extract_text(str(path), page_numbers=[index])
        except Exception as exc:  # pragma: no cover - runtime dependency
            logger.warning("pdfminer page extraction failed on page %d: %s", index + 1, exc)
            continue

        page_text = clean_ocr_artifacts((raw_text or "").strip())
        if not page_text:
            continue

        page_number = index + page_offset
        marked_text = f"{_format_page_marker(page_number)}\n{page_text}"
        pages.append(_PageData(marked_text, page_number, None, str(page_number)))
    return pages


def _extract_ocr(reader, path: Path, *, page_offset: int = 1, **_kw) -> list[_PageData]:
    """Strategy 4: OCR fallback (pytesseract)."""
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pytesseract and pdf2image are required for OCR PDF extraction.") from exc
    pages: list[_PageData] = []
    page_count = len(reader.pages)
    for index in range(1, page_count + 1):
        page_images = convert_from_path(str(path), first_page=index, last_page=index)
        if not page_images:
            continue
        page_text = clean_ocr_artifacts(
            (pytesseract.image_to_string(page_images[0]) or "").strip()
        )
        del page_images
        if not page_text:
            continue
        page_number = (index - 1) + page_offset
        marked_text = f"{_format_page_marker(page_number)}\n{page_text}"
        pages.append(_PageData(marked_text, page_number, None, str(page_number)))
    return pages


# Ordered extraction strategies for PDF ingestion
_PDF_STRATEGIES = [_extract_pymupdf, _extract_pypdf, _extract_pdfminer, _extract_ocr]


def ingest_pdf(
    file_path: str | Path,
    *,
    source_id: str,
    page_offset: int = 1,
) -> tuple[list[ParentChunk], list[ChildChunk]]:
    path = Path(file_path)
    if not source_id or not source_id.strip():
        raise ValueError("source_id must be a non-empty string.")
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    try:
        from pypdf import PdfReader
    except Exception as exc:  # pragma: no cover - dependency runtime
        raise RuntimeError("pypdf is required to ingest PDF files.") from exc

    reader = PdfReader(str(path))
    if not reader.pages:
        raise ValueError("PDF file has no pages.")

    for strategy in _PDF_STRATEGIES:
        pages = strategy(reader=reader, path=path, page_offset=page_offset)
        if pages:
            parents, children = _chunk_pages(pages, source_id)
            if parents:
                return parents, children

    raise ValueError("No extractable text found in PDF, even after OCR.")


def _coerce_embeddings(raw_embeddings: object) -> list[list[float]]:
    if hasattr(raw_embeddings, "tolist"):
        return raw_embeddings.tolist()
    if isinstance(raw_embeddings, list):
        return [list(map(float, emb)) for emb in raw_embeddings]
    raise TypeError("Unsupported embeddings type.")


def _context_snippet(text: str, start: int, end: int, radius: int = 120) -> str:
    lo = max(0, int(start) - radius)
    hi = min(len(text), int(end) + radius)
    snippet = text[lo:hi].strip()
    return re.sub(r"\s+", " ", snippet)


def _geotag_chunks(
    source_id: str,
    child_chunks: list[ChildChunk],
    storage: StorageEngine,
    *,
    ner_lists: Optional[list[list[dict[str, Any]]]] = None,
    ner_diagnostics: Optional["NERDiagnostics"] = None,
) -> "NERDiagnostics":
    """Extract place mentions with NER, geocode, and store grouped mention rows."""
    from .ner import NERDiagnostics

    diagnostics = ner_diagnostics

    try:
        import time
        import uuid

        from .config import (
            GEOTAG_FUZZY_THRESHOLD,
            GEOTAG_MIN_CONFIDENCE,
            GEOTAG_NER_THRESHOLD,
            GEOTAG_NER_CONTEXT_WINDOW,
        )
        from .geocoder import get_geocoder
        from .ner import extract_place_candidates_ner_with_diagnostics

        if ner_lists is None:
            texts = [chunk.text for chunk in child_chunks]
            ner_lists, diagnostics = extract_place_candidates_ner_with_diagnostics(
                texts,
                threshold=GEOTAG_NER_THRESHOLD,
                context_window_words=GEOTAG_NER_CONTEXT_WINDOW,
            )
        elif diagnostics is None:
            diagnostics = NERDiagnostics(ner_available=True, method="gliner")

        geocoder = get_geocoder()
        if not geocoder.is_available():
            logger.info("Geocoder cold during ingest geotagging; warming now for %s", source_id)
            geocoder.warm(background=False)
        if not geocoder.is_available():
            logger.warning("Geocoder unavailable after warm attempt — skipping geotagging for %s", source_id)
            return diagnostics or NERDiagnostics(ner_available=True, method="gliner")

        status = geocoder.status()
        version_info = status.get("version_info") if isinstance(status, dict) else None
        geocoder_version = "unknown"
        if isinstance(version_info, dict):
            geocoder_version = str(version_info.get("checksum") or version_info.get("build_timestamp") or "unknown")

        geocoded_at = time.time()

        mentions: list[dict] = []
        seen: set[tuple[int, str]] = set()

        # Flatten all candidates across all chunks so the geocoder can use
        # coherence across the whole document in one batch call
        flat_candidates: list[tuple[ChildChunk, dict]] = []
        for chunk, place_candidates in zip(child_chunks, ner_lists):
            for candidate in place_candidates:
                normalized = str(candidate.get("text", "")).strip()
                if normalized:
                    flat_candidates.append((chunk, candidate))

        place_names = [str(c.get("text", "")).strip() for _, c in flat_candidates]
        entity_types = [
            str(c.get("entity_type", "")).upper().strip() or None
            for _, c in flat_candidates
        ]

        batch_results = geocoder.forward_batch(
            place_names,
            threshold=GEOTAG_FUZZY_THRESHOLD,
            entity_types=entity_types,
        )

        for (chunk, candidate), match in zip(flat_candidates, batch_results):
            if match is None:
                continue
            if match.confidence < GEOTAG_MIN_CONFIDENCE:
                continue
            key = (match.place.geonameid, chunk.id)
            if key in seen:
                continue
            seen.add(key)

            normalized = str(candidate.get("text", "")).strip()
            entity_type = str(candidate.get("entity_type", "")).upper().strip() or None

            ner_score_raw = candidate.get("score", 0.0)
            try:
                ner_score = float(ner_score_raw)
            except (TypeError, ValueError):
                ner_score = 0.0

            mentions.append(
                {
                    "id": str(uuid.uuid4()),
                    "source_id": source_id,
                    "chunk_id": chunk.id,
                    "place_name": match.place.name,
                    "matched_input": normalized,
                    "geonameid": match.place.geonameid,
                    "lat": match.place.lat,
                    "lon": match.place.lon,
                    "confidence": round(match.confidence, 4),
                    "method": match.method.value,
                    "matched_on": match.matched_on,
                    "raw_score": round(float(match.score), 4),
                    "is_ambiguous": bool(match.ambiguous),
                    "candidate_count": int(match.candidate_count),
                    "margin_score": None if match.margin_score is None else round(float(match.margin_score), 4),
                    "entity_type": entity_type,
                    "ner_score": round(ner_score, 4),
                    "geocoder_version": geocoder_version,
                    "geocoded_at": geocoded_at,
                }
            )

        if mentions:
            storage.upsert_geo_mentions(mentions)
            logger.info("Geotagged %d mentions for source %s.", len(mentions), source_id)
        return diagnostics or NERDiagnostics(ner_available=True, method="gliner")
    except Exception as exc:
        logger.warning("Geotagging failed for %s (ingest unaffected): %s", source_id, exc)
        warning = f"Geotagging failed (ingest unaffected): {type(exc).__name__}: {exc}"
        if diagnostics is not None:
            prior = diagnostics.warning
            merged_warning = f"{prior}; {warning}" if prior else warning
            return NERDiagnostics(
                ner_available=diagnostics.ner_available,
                method=diagnostics.method,
                warning=merged_warning,
            )
        return NERDiagnostics(ner_available=False, method="empty", warning=warning)


def _peopletag_chunks(
    source_id: str,
    child_chunks: list[ChildChunk],
    storage: StorageEngine,
    *,
    ner_lists: Optional[list[list[dict[str, Any]]]] = None,
    ner_diagnostics: Optional["NERDiagnostics"] = None,
) -> "NERDiagnostics":
    """Extract person mentions, resolve canonical names, and persist mention rows."""
    from .ner import NERDiagnostics

    diagnostics = ner_diagnostics

    try:
        import uuid

        from .config import (
            PEOPLETAG_MIN_CONFIDENCE,
            PEOPLETAG_NER_CONTEXT_WINDOW,
            PEOPLETAG_NER_THRESHOLD,
        )
        from .ner import extract_person_candidates_ner_with_diagnostics
        from .person_resolver import get_person_resolver

        # Re-ingest guard: clear prior mentions for this source before reprocessing.
        storage.delete_person_mentions_by_source(source_id)

        resolver = get_person_resolver()
        resolver.warm_from_rows(storage.list_person_mentions_for_registry())

        if ner_lists is None:
            texts = [chunk.text for chunk in child_chunks]
            ner_lists, diagnostics = extract_person_candidates_ner_with_diagnostics(
                texts,
                threshold=PEOPLETAG_NER_THRESHOLD,
                context_window_words=PEOPLETAG_NER_CONTEXT_WINDOW,
            )
        elif diagnostics is None:
            diagnostics = NERDiagnostics(ner_available=True, method="gliner")

        mentions: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()

        for chunk, candidates in zip(child_chunks, ner_lists):
            for candidate in candidates:
                raw_name = str(candidate.get("text", "")).strip()
                if not raw_name:
                    continue

                try:
                    ner_score = float(candidate.get("score", 0.0))
                except (TypeError, ValueError):
                    ner_score = 0.0
                if ner_score < PEOPLETAG_MIN_CONFIDENCE:
                    continue

                try:
                    start = int(candidate.get("start", 0))
                except (TypeError, ValueError):
                    start = 0
                try:
                    end = int(candidate.get("end", start + len(raw_name)))
                except (TypeError, ValueError):
                    end = start + len(raw_name)
                start = max(0, start)
                end = max(start, min(len(chunk.text), end))

                context_words = [
                    word.strip()
                    for word in candidate.get("context_words", [])
                    if isinstance(word, str) and word.strip()
                ]
                snippet = _context_snippet(chunk.text, start, end)

                key = (source_id, raw_name.lower(), snippet)
                if key in seen:
                    continue
                seen.add(key)

                resolved = resolver.resolve(
                    raw_name=raw_name,
                    source_id=source_id,
                    ner_score=ner_score,
                    context_words=context_words,
                    context_snippet=snippet,
                )
                if not resolved:
                    continue

                confidence = float(resolved.get("confidence", 0.0))
                if confidence < PEOPLETAG_MIN_CONFIDENCE:
                    continue

                mentions.append(
                    {
                        "id": str(uuid.uuid4()),
                        "source_id": source_id,
                        "chunk_id": chunk.id,
                        "raw_name": str(resolved.get("raw_name", raw_name)),
                        "canonical_name": str(resolved.get("canonical_name", raw_name)),
                        "confidence": confidence,
                        "method": str(resolved.get("method", "new")),
                        "role_hint": resolved.get("role_hint"),
                        "context_snippet": str(resolved.get("context_snippet", snippet)),
                    }
                )

        if mentions:
            storage.upsert_person_mentions(mentions)
            logger.info("Peopletagged %d mentions for source %s.", len(mentions), source_id)
        return diagnostics or NERDiagnostics(ner_available=True, method="gliner")
    except Exception as exc:
        logger.warning("Peopletagging failed for %s (ingest unaffected): %s", source_id, exc)
        warning = f"Peopletagging failed (ingest unaffected): {type(exc).__name__}: {exc}"
        if diagnostics is not None:
            prior = diagnostics.warning
            merged_warning = f"{prior}; {warning}" if prior else warning
            return NERDiagnostics(
                ner_available=diagnostics.ner_available,
                method=diagnostics.method,
                warning=merged_warning,
            )
        return NERDiagnostics(ner_available=False, method="empty", warning=warning)


def ingest_file_to_storage(
    file_path: str | Path,
    *,
    source_id: str,
    page_number: Optional[int],
    storage: StorageEngine,
    embedding_model: object,
    summarize: bool = False,
    summary_generator: Optional[MlxGenerator] = None,
    geotag: bool = False,
    peopletag: bool = False,
    citation_reference: Optional[str] = None,
    page_offset: int = 1,
    tracer: Optional[Any] = None,
) -> tuple[int, int, IngestNERDiagnostics]:
    path = Path(file_path)
    suffix = path.suffix.lower()
    with start_span(
        tracer,
        "rag.ingest.file",
        span_kind=SPAN_KIND_CHAIN,
        attributes={
            "input.value": source_id,
            "rag.ingest.file_path": str(path),
            "rag.ingest.file_type": suffix,
            "rag.ingest.page_number": page_number,
            "rag.ingest.page_offset": page_offset,
            "rag.ingest.summarize": summarize,
            "rag.ingest.geotag": geotag,
            "rag.ingest.peopletag": peopletag,
            "rag.ingest.citation_reference": citation_reference,
        },
    ) as ingest_span:
        try:
            with start_span(
                tracer,
                "rag.ingest.parse_and_chunk",
                span_kind=SPAN_KIND_CHAIN,
                attributes={
                    "rag.ingest.file_type": suffix,
                },
            ) as parse_span:
                if suffix == ".pdf":
                    parents, children = ingest_pdf(path, source_id=source_id, page_offset=page_offset)
                elif suffix in {".md", ".markdown"}:
                    parents, children = ingest_markdown(
                        path,
                        source_id=source_id,
                        page_number=page_number,
                    )
                else:
                    raise ValueError(f"Unsupported file type: {path.suffix}")
                set_span_attributes(
                    parse_span,
                    {
                        "rag.ingest.parents_count": len(parents),
                        "rag.ingest.children_count": len(children),
                    },
                )

            texts = [child.text for child in children]
            with start_span(
                tracer,
                "rag.ingest.embed_children",
                span_kind=SPAN_KIND_EMBEDDING,
                attributes={
                    "rag.ingest.children_count": len(children),
                },
            ) as embed_span:
                try:
                    embeddings = embedding_model.encode(texts, normalize_embeddings=True)
                except Exception as exc:  # pragma: no cover - dependency runtime
                    mark_span_error(embed_span, f"{type(exc).__name__}: {exc}")
                    raise RuntimeError("Embedding model encode failed.") from exc
                set_span_attributes(
                    embed_span,
                    {
                        "rag.ingest.children_encoded_count": len(texts),
                    },
                )

            with start_span(
                tracer,
                "rag.ingest.storage.write_parents",
                span_kind=SPAN_KIND_CHAIN,
                attributes={"rag.ingest.parents_count": len(parents)},
            ):
                storage.add_parents(parents)

            with start_span(
                tracer,
                "rag.ingest.storage.write_children",
                span_kind=SPAN_KIND_CHAIN,
                attributes={"rag.ingest.children_count": len(children)},
            ) as children_span:
                try:
                    storage.add_children(children, embeddings=_coerce_embeddings(embeddings))
                except Exception as exc:
                    mark_span_error(children_span, f"{type(exc).__name__}: {exc}")
                    # Roll back the parents we just wrote so storage stays consistent.
                    logger.error("Child chunk write failed for source '%s'; rolling back parents", source_id)
                    try:
                        storage.delete_source(source_id)
                    except Exception:
                        logger.exception("Rollback delete_source('%s') also failed", source_id)
                    raise

            if geotag and peopletag:
                from .config import (
                    GEOTAG_NER_CONTEXT_WINDOW,
                    GEOTAG_NER_THRESHOLD,
                    PEOPLETAG_NER_CONTEXT_WINDOW,
                    PEOPLETAG_NER_THRESHOLD,
                )
                from .ner import extract_place_and_person_candidates_ner_with_diagnostics

                texts = [child.text for child in children]
                (
                    place_candidates,
                    person_candidates,
                    geotag_diag,
                    peopletag_diag,
                ) = extract_place_and_person_candidates_ner_with_diagnostics(
                    texts,
                    geo_threshold=GEOTAG_NER_THRESHOLD,
                    people_threshold=PEOPLETAG_NER_THRESHOLD,
                    geo_context_window_words=GEOTAG_NER_CONTEXT_WINDOW,
                    people_context_window_words=PEOPLETAG_NER_CONTEXT_WINDOW,
                )
                geotag_diag = _geotag_chunks(
                    source_id,
                    children,
                    storage,
                    ner_lists=place_candidates,
                    ner_diagnostics=geotag_diag,
                )
                peopletag_diag = _peopletag_chunks(
                    source_id,
                    children,
                    storage,
                    ner_lists=person_candidates,
                    ner_diagnostics=peopletag_diag,
                )
                ingest_ner_diagnostics = IngestNERDiagnostics(
                    geotag_ner=geotag_diag,
                    peopletag_ner=peopletag_diag,
                )
            else:
                geotag_diag = None
                peopletag_diag = None
                if geotag:
                    geotag_diag = _geotag_chunks(source_id, children, storage)
                if peopletag:
                    peopletag_diag = _peopletag_chunks(source_id, children, storage)
                ingest_ner_diagnostics = IngestNERDiagnostics(
                    geotag_ner=geotag_diag,
                    peopletag_ner=peopletag_diag,
                )

            set_span_attributes(
                ingest_span,
                {
                    "rag.ingest.ner.geotag_method": (
                        ingest_ner_diagnostics.geotag_ner.method
                        if ingest_ner_diagnostics.geotag_ner is not None
                        else "disabled"
                    ),
                    "rag.ingest.ner.peopletag_method": (
                        ingest_ner_diagnostics.peopletag_ner.method
                        if ingest_ner_diagnostics.peopletag_ner is not None
                        else "disabled"
                    ),
                },
            )

            if summarize:
                generator = summary_generator
                if generator is None:
                    raise ValueError("summary_generator is required when summarize=True")
                with start_span(
                    tracer,
                    "rag.ingest.summary.generate",
                    span_kind=SPAN_KIND_LLM,
                    attributes={
                        "rag.ingest.parents_count": len(parents),
                        "rag.ingest.summary_context_limit": _SUMMARY_CONTEXT_CHAR_LIMIT,
                    },
                ) as summary_span:
                    context = "\n\n".join(parent.text for parent in parents)
                    context = _sample_context(context, _SUMMARY_CONTEXT_CHAR_LIMIT)
                    messages = build_ingest_summary_messages(context)
                    summary = generator.generate_chat(messages)
                    storage.upsert_source_summary(
                        source_id=source_id,
                        summary=summary,
                        citation_reference=citation_reference,
                        page_offset=page_offset,
                    )
                    set_span_attributes(
                        summary_span,
                        {
                            "rag.prompt.message_count": len(messages),
                            "rag.context_chars": len(context),
                            "rag.output_chars": len(summary),
                            "output.value": summary,
                        },
                    )

            with start_span(
                tracer,
                "rag.ingest.persist_metadata",
                span_kind=SPAN_KIND_CHAIN,
                attributes={"rag.ingest.page_offset": page_offset},
            ):
                storage.persist_source_page_offset(
                    source_id,
                    page_offset,
                    citation_reference=citation_reference,
                )

            set_span_attributes(
                ingest_span,
                {
                    "rag.ingest.parents_count": len(parents),
                    "rag.ingest.children_count": len(children),
                    "output.value": f"{source_id}: {len(parents)} parents, {len(children)} children",
                },
            )
            return len(parents), len(children), ingest_ner_diagnostics
        except Exception as exc:
            mark_span_error(ingest_span, f"{type(exc).__name__}: {exc}")
            raise
