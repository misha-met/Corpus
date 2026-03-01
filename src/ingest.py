from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from .models import ChildChunk, Metadata, ParentChunk
from .generation import build_ingest_summary_messages
from .generator import MlxGenerator
from .storage import StorageEngine

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

CHILD_MIN_TOKENS = 200
CHILD_MAX_TOKENS = 300
CHILD_TARGET_TOKENS = 250
CHILD_OVERLAP_TOKENS = 50
CHILD_OVERLAP_SENTENCES = 2

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
CLAUSE_BOUNDARY_RE = re.compile(r";\s+|,\s+(?:and|but|which|or)\s+", re.IGNORECASE)


@dataclass(frozen=True)
class _Section:
    header_path: str
    text: str


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
    if _token_count(sentence) <= max(1, int(target_tokens * 0.4)):
        return [sentence]

    boundary_matches = list(CLAUSE_BOUNDARY_RE.finditer(sentence))
    if not boundary_matches:
        return [sentence]

    midpoint = len(sentence) / 2
    best = min(boundary_matches, key=lambda match: abs(match.start() - midpoint))
    split_index = best.start() + 1 if sentence[best.start()] in {",", ";"} else best.start()

    left = sentence[:split_index].strip()
    right = sentence[split_index:].strip()
    if not left or not right:
        return [sentence]

    return [left, right]


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
        metadata = Metadata(
            source_id=source_id,
            page_number=page_number,
            page_label=page_label,
            display_page=display_page,
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

    for chunk_sentences in sentence_chunks:
        text = " ".join(chunk_sentences).strip()
        token_count = _token_count(text)
        if token_count < CHILD_MIN_TOKENS:
            # Instead of dropping, merge into previous child or create sole child
            if child_chunks:
                # Merge into previous child's text
                prev = child_chunks[-1]
                merged_text = prev.text + " " + text
                # Re-create with same id and metadata (frozen model)
                child_chunks[-1] = ChildChunk.model_construct(
                    id=prev.id, text=merged_text, metadata=prev.metadata,
                )
                _merge_count += 1
            else:
                # No previous child — create a sole child (short but searchable)
                metadata = Metadata(
                    source_id=parent.metadata.source_id,
                    page_number=parent.metadata.page_number,
                    page_label=parent.metadata.page_label,
                    display_page=parent.metadata.display_page,
                    header_path=parent.metadata.header_path,
                    parent_id=parent.id,
                )
                child_chunks.append(ChildChunk(text=text, metadata=metadata))
                _sole_child_count += 1
            continue

        metadata = Metadata(
            source_id=parent.metadata.source_id,
            page_number=parent.metadata.page_number,
            page_label=parent.metadata.page_label,
            display_page=parent.metadata.display_page,
            header_path=parent.metadata.header_path,
            parent_id=parent.id,
        )
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


def _extract_pypdf(reader, **_kw) -> list[_PageData]:
    """Strategy 1: pypdf per-page extraction."""
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
        display_page = page_label if page_label else str(index)
        pages.append(_PageData(page_text, index, page_label, display_page))
    return pages


def _extract_pdfminer(path: Path, **_kw) -> list[_PageData]:
    """Strategy 2: pdfminer whole-document fallback."""
    try:
        from pdfminer.high_level import extract_text
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pdfminer.six is required for fallback PDF extraction.") from exc
    fallback_text = clean_ocr_artifacts((extract_text(str(path)) or "").strip())
    if fallback_text:
        return [_PageData(fallback_text, None, None, None)]
    return []


def _extract_pymupdf(path: Path, **_kw) -> list[_PageData]:
    """Strategy 3: PyMuPDF per-page fallback."""
    try:
        import fitz
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyMuPDF is required for additional PDF extraction fallback.") from exc
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
        page_number = index + 1
        display_page = page_label if page_label else str(page_number)
        pages.append(_PageData(page_text, page_number, page_label, display_page))
    return pages


def _extract_ocr(reader, path: Path, **_kw) -> list[_PageData]:
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
        pages.append(_PageData(page_text, index, None, str(index)))
    return pages


# Ordered extraction strategies for PDF ingestion
_PDF_STRATEGIES = [_extract_pypdf, _extract_pdfminer, _extract_pymupdf, _extract_ocr]


def ingest_pdf(
    file_path: str | Path,
    *,
    source_id: str,
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
        pages = strategy(reader=reader, path=path)
        if pages:
            parents, children = _chunk_pages(pages, source_id)
            if parents:
                return parents, children

    raise ValueError("No extractable text found in PDF, even after OCR.")
    if not children:
        raise ValueError("No child chunks produced from PDF content.")

    return parents, children


def _coerce_embeddings(raw_embeddings: object) -> list[list[float]]:
    if hasattr(raw_embeddings, "tolist"):
        return raw_embeddings.tolist()
    if isinstance(raw_embeddings, list):
        return [list(map(float, emb)) for emb in raw_embeddings]
    raise TypeError("Unsupported embeddings type.")


def ingest_file_to_storage(
    file_path: str | Path,
    *,
    source_id: str,
    page_number: Optional[int],
    storage: StorageEngine,
    embedding_model: object,
    summarize: bool = False,
    summary_generator: Optional[MlxGenerator] = None,
) -> tuple[int, int]:
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        parents, children = ingest_pdf(path, source_id=source_id)
    elif suffix in {".md", ".markdown"}:
        parents, children = ingest_markdown(
            path,
            source_id=source_id,
            page_number=page_number,
        )
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    texts = [child.text for child in children]
    try:
        embeddings = embedding_model.encode(texts, normalize_embeddings=True)
    except Exception as exc:  # pragma: no cover - dependency runtime
        raise RuntimeError("Embedding model encode failed.") from exc

    storage.add_parents(parents)
    try:
        storage.add_children(children, embeddings=_coerce_embeddings(embeddings))
    except Exception:
        # Roll back the parents we just wrote so storage stays consistent.
        logger.error("Child chunk write failed for source '%s'; rolling back parents", source_id)
        try:
            storage.delete_source(source_id)
        except Exception:
            logger.exception("Rollback delete_source('%s') also failed", source_id)
        raise

    if summarize:
        generator = summary_generator
        if generator is None:
            raise ValueError("summary_generator is required when summarize=True")
        context = "\n\n".join(parent.text for parent in parents)
        context = _sample_context(context, _SUMMARY_CONTEXT_CHAR_LIMIT)
        messages = build_ingest_summary_messages(context)
        summary = generator.generate_chat(messages)
        storage.upsert_source_summary(source_id=source_id, summary=summary)

    return len(parents), len(children)
