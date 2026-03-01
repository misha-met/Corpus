"""Text snapshot storage for source documents.

Stores plain-text snapshots of ingested documents under ``data/source_cache/``.
These snapshots serve as fallback content when the original file is moved or
deleted, enabling the ``/content`` endpoint to always serve document text.

Resolution order for ``/content``:
1. Original source_path (if file still exists and is readable)
2. Cached snapshot_path (always available after ingest)
3. 404 (source was deleted from both locations)
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "source_cache"


def _sanitize_filename(source_id: str) -> str:
    """Convert a source_id to a safe filename.

    Uses the source_id directly if it's filesystem-safe, otherwise
    falls back to a hash-based name.
    """
    safe_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.")
    if all(c in safe_chars for c in source_id) and len(source_id) <= 200:
        return f"{source_id}.txt"
    # Hash-based fallback for source IDs with special characters
    h = hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:16]
    return f"{h}.txt"


def save_snapshot(
    source_id: str,
    text: str,
    *,
    cache_dir: Optional[Path] = None,
) -> str:
    """Save a text snapshot and return the path.

    Parameters
    ----------
    source_id : str
        Source identifier (used to derive filename).
    text : str
        Full text content to cache.
    cache_dir : Path | None
        Override the default cache directory.

    Returns
    -------
    str
        Path to the saved snapshot file (relative to project root).
    """
    directory = cache_dir or DEFAULT_CACHE_DIR
    directory.mkdir(parents=True, exist_ok=True)

    filename = _sanitize_filename(source_id)
    filepath = directory / filename
    filepath.write_text(text, encoding="utf-8")
    logger.info("Saved text snapshot: %s (%d chars)", filepath, len(text))
    return str(filepath)


def read_snapshot(snapshot_path: str) -> Optional[str]:
    """Read a cached text snapshot.

    Returns None if the file doesn't exist or can't be read.
    """
    path = Path(snapshot_path)
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to read snapshot %s: %s", snapshot_path, exc)
        return None


def delete_snapshot(snapshot_path: str) -> bool:
    """Delete a cached text snapshot.

    Returns True if the file was deleted, False if it didn't exist.
    """
    path = Path(snapshot_path)
    if not path.is_file():
        return False
    try:
        path.unlink()
        logger.info("Deleted snapshot: %s", snapshot_path)
        return True
    except Exception as exc:
        logger.warning("Failed to delete snapshot %s: %s", snapshot_path, exc)
        return False


def read_original_file(source_path: str) -> Optional[str]:
    """Read the original source file as text.

    Returns None if the file doesn't exist or can't be read as text.
    Handles PDF files by extracting text, and reads other files as plain text.
    """
    path = Path(source_path)
    if not path.is_file():
        return None

    try:
        # PDF files need special handling
        if path.suffix.lower() == ".pdf":
            return _extract_pdf_text(path)
        # All other files: read as UTF-8 text
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to read original file %s: %s", source_path, exc)
        return None


def _extract_pdf_text(path: Path) -> Optional[str]:
    """Extract text from a PDF file using PyMuPDF (fitz)."""
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(str(path))
        pages = []
        for page in doc:
            text = page.get_text()
            if text.strip():
                pages.append(text)
        doc.close()
        return "\n\n".join(pages) if pages else None
    except ImportError:
        logger.warning("PyMuPDF not installed; cannot extract PDF text from %s", path)
        return None
    except Exception as exc:
        logger.warning("PDF text extraction failed for %s: %s", path, exc)
        return None


def resolve_content(
    source_path: Optional[str],
    snapshot_path: Optional[str],
) -> Optional[tuple[str, str]]:
    """Resolve document content using the fallback chain.

    Resolution order:
    1. Original source_path (if readable)
    2. Cached snapshot_path (if exists)

    Returns
    -------
    tuple[str, str] | None
        (content, source_type) where source_type is 'original' or 'snapshot'.
        Returns None if neither is available.
    """
    # Try original file first
    if source_path:
        content = read_original_file(source_path)
        if content:
            return (content, "original")

    # Try cached snapshot
    if snapshot_path:
        content = read_snapshot(snapshot_path)
        if content:
            return (content, "snapshot")

    return None
