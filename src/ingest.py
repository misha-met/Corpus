from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from .models import ChildChunk, Metadata, ParentChunk
from .intent import Intent
from .generation import build_prompt
from .generator import MlxGenerator
from .storage import StorageEngine

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
            page_label=parent.metadata.page_label,
            display_page=parent.metadata.display_page,
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

    # For markdown, display_page is simply str(page_number) if provided
    display_page = str(page_number) if page_number is not None else None

    parents: list[ParentChunk] = []
    children: list[ChildChunk] = []

    for section in sections:
        for parent in _split_parent_chunks(
            section,
            source_id=source_id.strip(),
            page_number=page_number,
            page_label=None,  # No page labels in markdown
            display_page=display_page,
        ):
            parents.append(parent)
            children.extend(_split_child_chunks(parent))

    if not parents:
        raise ValueError("No parent chunks produced from markdown content.")
    if not children:
        raise ValueError("No child chunks produced from markdown content.")

    return parents, children


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

    parents: list[ParentChunk] = []
    children: list[ChildChunk] = []

    for index, page in enumerate(reader.pages, start=1):
        page_text = clean_ocr_artifacts((page.extract_text() or "").strip())
        if not page_text:
            continue
        
        # Extract logical page label (e.g., 'iii', 'xii', '1', '2')
        # pypdf provides page labels through page.get_label() or reader.page_labels
        page_label: Optional[str] = None
        try:
            # Try to get the label from the reader's page_labels dictionary
            if hasattr(reader, 'page_labels') and reader.page_labels:
                page_label = reader.page_labels.get(index - 1)  # 0-indexed
            # Fallback: try page.get_label() if available (some pypdf versions)
            if page_label is None and hasattr(page, 'get_label'):
                page_label = page.get_label()
        except Exception:
            pass  # Page label extraction is best-effort
        
        # display_page: prefer page_label, fallback to str(page_number)
        display_page = page_label if page_label else str(index)
        
        section = _Section(header_path="Document", text=page_text)
        for parent in _split_parent_chunks(
            section,
            source_id=source_id.strip(),
            page_number=index,
            page_label=page_label,
            display_page=display_page,
        ):
            parents.append(parent)
            children.extend(_split_child_chunks(parent))

    if not parents:
        try:
            from pdfminer.high_level import extract_text
        except Exception as exc:  # pragma: no cover - dependency runtime
            raise RuntimeError(
                "pdfminer.six is required for fallback PDF extraction."
            ) from exc

        fallback_text = clean_ocr_artifacts((extract_text(str(path)) or "").strip())
        if fallback_text:
            # pdfminer fallback: no page-level info available
            section = _Section(header_path="Document", text=fallback_text)
            for parent in _split_parent_chunks(
                section,
                source_id=source_id.strip(),
                page_number=None,
                page_label=None,
                display_page=None,
            ):
                parents.append(parent)
                children.extend(_split_child_chunks(parent))

    if not parents:
        try:
            import fitz  # PyMuPDF
        except Exception as exc:  # pragma: no cover - dependency runtime
            raise RuntimeError(
                "PyMuPDF is required for additional PDF extraction fallback."
            ) from exc

        doc = fitz.open(str(path))
        for index in range(doc.page_count):
            page = doc.load_page(index)
            page_text = clean_ocr_artifacts((page.get_text("text") or "").strip())
            if not page_text:
                continue
            
            # PyMuPDF: extract page label if available
            page_label: Optional[str] = None
            try:
                # fitz stores page labels in doc.get_page_labels() or page.get_label()
                if hasattr(doc, 'get_page_labels'):
                    labels = doc.get_page_labels()
                    if labels and index < len(labels):
                        page_label = labels[index]
            except Exception:
                pass  # Page label extraction is best-effort
            
            page_number = index + 1
            display_page = page_label if page_label else str(page_number)
            
            section = _Section(header_path="Document", text=page_text)
            for parent in _split_parent_chunks(
                section,
                source_id=source_id.strip(),
                page_number=page_number,
                page_label=page_label,
                display_page=display_page,
            ):
                parents.append(parent)
                children.extend(_split_child_chunks(parent))

    if not parents:
        try:
            from pdf2image import convert_from_path
            import pytesseract
        except Exception as exc:  # pragma: no cover - dependency runtime
            raise RuntimeError(
                "pytesseract and pdf2image are required for OCR PDF extraction."
            ) from exc

        images = convert_from_path(str(path))
        for index, image in enumerate(images, start=1):
            page_text = clean_ocr_artifacts(
                (pytesseract.image_to_string(image) or "").strip()
            )
            if not page_text:
                continue
            # OCR fallback: no page labels available from image-based extraction
            section = _Section(header_path="Document", text=page_text)
            for parent in _split_parent_chunks(
                section,
                source_id=source_id.strip(),
                page_number=index,
                page_label=None,
                display_page=str(index),
            ):
                parents.append(parent)
                children.extend(_split_child_chunks(parent))

    if not parents:
        raise ValueError(
            "No extractable text found in PDF, even after OCR."
        )
    if not children:
        raise ValueError("No child chunks produced from PDF content.")

    return parents, children


def _coerce_embeddings(raw_embeddings: object) -> list[list[float]]:
    if hasattr(raw_embeddings, "tolist"):
        return raw_embeddings.tolist()
    if isinstance(raw_embeddings, list):
        return [list(map(float, emb)) for emb in raw_embeddings]
    raise TypeError("Unsupported embeddings type.")


def ingest_markdown_to_storage(
    file_path: str | Path,
    *,
    source_id: str,
    page_number: Optional[int],
    storage: StorageEngine,
    embedding_model: object,
    bm25_path: Optional[Path] = None,
) -> tuple[int, int]:
    parents, children = ingest_markdown(
        file_path,
        source_id=source_id,
        page_number=page_number,
    )

    texts = [child.text for child in children]
    try:
        embeddings = embedding_model.encode(texts, normalize_embeddings=True)
    except Exception as exc:  # pragma: no cover - dependency runtime
        raise RuntimeError("Embedding model encode failed.") from exc

    storage.add_parents(parents)
    storage.add_children(children, embeddings=_coerce_embeddings(embeddings))

    if bm25_path is not None:
        storage.persist_bm25(bm25_path)

    return len(parents), len(children)


def ingest_file_to_storage(
    file_path: str | Path,
    *,
    source_id: str,
    page_number: Optional[int],
    storage: StorageEngine,
    embedding_model: object,
    bm25_path: Optional[Path] = None,
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
    storage.add_children(children, embeddings=_coerce_embeddings(embeddings))

    if bm25_path is not None:
        storage.persist_bm25(bm25_path)

    if summarize:
        generator = summary_generator
        if generator is None:
            raise ValueError("summary_generator is required when summarize=True")
        context = "\n\n".join(parent.text for parent in parents)
        if len(context) > 12000:
            context = context[:12000]
        prompt = build_prompt(
            context=context,
            question="Summarize this document.",
            intent=Intent.SUMMARIZE,
        )
        summary = generator.generate(prompt)
        storage.upsert_source_summary(source_id=source_id, summary=summary)

    return len(parents), len(children)
