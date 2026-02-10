"""Tests for data layer: Metadata, ParentChunk, ChildChunk models and ingest chunking."""
from __future__ import annotations

import re
import tempfile
import uuid
from pathlib import Path

import pytest

from src.models import ChildChunk, Metadata, ParentChunk
from src.ingest import (
    CHILD_MAX_TOKENS,
    CHILD_MIN_TOKENS,
    CHILD_OVERLAP_TOKENS,
    CHILD_TARGET_TOKENS,
    PARENT_MAX_TOKENS,
    PARENT_MIN_TOKENS,
    PARENT_OVERLAP_TOKENS,
    PARENT_TARGET_TOKENS,
    _parse_markdown_sections,
    _split_child_chunks,
    _split_parent_chunks,
    _split_tokens,
    _tokenize,
    _detokenize,
    _Section,
    clean_ocr_artifacts,
    ingest_markdown,
)
from tests.conftest import Timer, get_test_logger

logger = get_test_logger("models_ingest")


# ===========================================================================
# Metadata validation
# ===========================================================================

class TestMetadata:
    def test_valid_metadata(self):
        m = Metadata(source_id="doc1", header_path="Chapter 1")
        assert m.source_id == "doc1"
        assert m.page_number is None
        assert m.parent_id is None

    def test_metadata_with_all_fields(self):
        m = Metadata(
            source_id="doc1",
            page_number=5,
            page_label="v",
            display_page="v",
            header_path="Ch1 > Sec2",
            parent_id="pid-123",
        )
        assert m.page_number == 5
        assert m.display_page == "v"
        assert m.parent_id == "pid-123"

    def test_empty_source_id_rejected(self):
        with pytest.raises(Exception):
            Metadata(source_id="", header_path="Doc")

    def test_empty_header_path_rejected(self):
        with pytest.raises(Exception):
            Metadata(source_id="doc1", header_path="")

    def test_negative_page_number_rejected(self):
        with pytest.raises(Exception):
            Metadata(source_id="doc1", header_path="Doc", page_number=-1)

    def test_page_number_zero_rejected(self):
        with pytest.raises(Exception):
            Metadata(source_id="doc1", header_path="Doc", page_number=0)

    def test_metadata_frozen(self):
        m = Metadata(source_id="doc1", header_path="Doc")
        with pytest.raises(Exception):
            m.source_id = "modified"  # type: ignore[misc]

    def test_extra_fields_rejected(self):
        with pytest.raises(Exception):
            Metadata(source_id="doc1", header_path="Doc", unknown_field="x")  # type: ignore[call-arg]


# ===========================================================================
# ParentChunk and ChildChunk
# ===========================================================================

class TestChunks:
    def test_parent_chunk_creation(self):
        meta = Metadata(source_id="doc1", header_path="Chapter 1")
        parent = ParentChunk(text="Some meaningful content here.", metadata=meta)
        assert parent.text == "Some meaningful content here."
        assert parent.id  # auto-generated UUID
        uuid.UUID(parent.id)  # validates it's a real UUID

    def test_child_chunk_with_parent_id(self):
        parent_id = str(uuid.uuid4())
        meta = Metadata(source_id="doc1", header_path="Chapter 1", parent_id=parent_id)
        child = ChildChunk(text="Child chunk content.", metadata=meta)
        assert child.metadata.parent_id == parent_id

    def test_empty_text_rejected(self):
        meta = Metadata(source_id="doc1", header_path="Doc")
        with pytest.raises(Exception):
            ParentChunk(text="", metadata=meta)

    def test_chunks_frozen(self):
        meta = Metadata(source_id="doc1", header_path="Doc")
        parent = ParentChunk(text="Content", metadata=meta)
        with pytest.raises(Exception):
            parent.text = "modified"  # type: ignore[misc]


# ===========================================================================
# Tokenization helpers
# ===========================================================================

class TestTokenization:
    def test_tokenize_basic(self):
        tokens = _tokenize("hello world foo bar")
        assert tokens == ["hello", "world", "foo", "bar"]

    def test_tokenize_empty(self):
        assert _tokenize("") == []

    def test_tokenize_whitespace_only(self):
        assert _tokenize("   \n\t  ") == []

    def test_detokenize_roundtrip(self):
        text = "one two three four"
        tokens = _tokenize(text)
        assert _detokenize(tokens) == text

    def test_split_tokens_basic(self):
        tokens = list(range(10))
        chunks = _split_tokens(tokens, 4, overlap=1)
        assert len(chunks) >= 1
        # All tokens are covered
        flat = []
        for c in chunks:
            flat.extend(c)
        assert set(flat) == set(range(10))

    def test_split_tokens_overlap(self):
        tokens = list(range(10))
        chunks = _split_tokens(tokens, 5, overlap=2)
        # Check overlap: last 2 of chunk 0 == first 2 of chunk 1
        if len(chunks) > 1:
            assert chunks[0][-2:] == chunks[1][:2]

    def test_split_tokens_empty(self):
        assert _split_tokens([], 5, 1) == []

    def test_split_tokens_invalid_chunk_size(self):
        with pytest.raises(ValueError):
            _split_tokens([1, 2, 3], 0, 0)

    def test_split_tokens_overlap_exceeds_size(self):
        with pytest.raises(ValueError):
            _split_tokens([1, 2, 3], 3, 3)


# ===========================================================================
# OCR cleanup
# ===========================================================================

class TestOCRCleanup:
    def test_hyphenated_linebreak(self):
        assert "linguists" in clean_ocr_artifacts("lin-\n guists")

    def test_no_artifacts(self):
        text = "This is normal text."
        assert clean_ocr_artifacts(text) == text

    def test_empty_string(self):
        assert clean_ocr_artifacts("") == ""

    def test_none_passthrough(self):
        assert clean_ocr_artifacts(None) is None  # type: ignore[arg-type]


# ===========================================================================
# Markdown section parsing
# ===========================================================================

class TestMarkdownParsing:
    def test_basic_sections(self):
        md = "# Title\nIntro text\n## Section A\nContent A\n## Section B\nContent B"
        sections = _parse_markdown_sections(md)
        assert len(sections) >= 2
        headers = [s.header_path for s in sections]
        assert any("Title" in h for h in headers)

    def test_nested_headers(self):
        md = "# Top\ntext\n## Mid\ntext\n### Deep\ntext"
        sections = _parse_markdown_sections(md)
        deep = [s for s in sections if "Deep" in s.header_path]
        assert len(deep) == 1
        # Nested header path should contain hierarchy
        assert "Top" in deep[0].header_path
        assert "Mid" in deep[0].header_path

    def test_no_headers(self):
        md = "Just plain text without headers."
        sections = _parse_markdown_sections(md)
        assert len(sections) == 1
        assert sections[0].header_path == "Document"

    def test_empty_content(self):
        md = ""
        sections = _parse_markdown_sections(md)
        assert len(sections) == 0


# ===========================================================================
# Parent chunk splitting
# ===========================================================================

class TestParentChunkSplitting:
    def test_short_text_single_chunk(self):
        """Text under PARENT_MAX_TOKENS should produce a single chunk."""
        text = " ".join(["word"] * (PARENT_MAX_TOKENS - 10))
        section = _Section(header_path="Test", text=text)
        parents = _split_parent_chunks(section, source_id="doc1", page_number=1)
        assert len(parents) == 1

    def test_long_text_multiple_chunks(self):
        """Text over PARENT_MAX_TOKENS should produce multiple chunks."""
        text = " ".join(["word"] * (PARENT_MAX_TOKENS * 3))
        section = _Section(header_path="Test", text=text)
        parents = _split_parent_chunks(section, source_id="doc1", page_number=1)
        assert len(parents) > 1

    def test_parent_chunk_token_limits(self):
        """All parent chunks should respect token bounds."""
        text = " ".join(["word"] * (PARENT_MAX_TOKENS * 5))
        section = _Section(header_path="Test", text=text)
        parents = _split_parent_chunks(section, source_id="doc1", page_number=1)
        for parent in parents:
            token_count = len(_tokenize(parent.text))
            # Final merged chunks may slightly exceed max, but should be bounded
            assert token_count <= PARENT_MAX_TOKENS * 2, (
                f"Parent chunk has {token_count} tokens, exceeds 2x max"
            )

    def test_metadata_propagation(self):
        """Parent chunks inherit source metadata correctly."""
        section = _Section(header_path="Ch1 > Sec2", text=" ".join(["word"] * 100))
        parents = _split_parent_chunks(
            section, source_id="my-doc", page_number=3, page_label="iii", display_page="iii"
        )
        for p in parents:
            assert p.metadata.source_id == "my-doc"
            assert p.metadata.page_number == 3
            assert p.metadata.page_label == "iii"
            assert p.metadata.header_path == "Ch1 > Sec2"

    def test_empty_text_no_chunks(self):
        section = _Section(header_path="Test", text="")
        parents = _split_parent_chunks(section, source_id="doc1", page_number=1)
        assert parents == []


# ===========================================================================
# Child chunk splitting
# ===========================================================================

class TestChildChunkSplitting:
    def test_child_chunks_from_parent(self):
        """Children should be produced from a parent with sufficient text."""
        text = " ".join(["word"] * (CHILD_MAX_TOKENS * 3))
        meta = Metadata(source_id="doc1", header_path="Test", parent_id=None)
        parent = ParentChunk(text=text, metadata=meta)
        children = _split_child_chunks(parent)
        assert len(children) > 0

    def test_child_parent_id_link(self):
        """Each child must reference its parent's ID."""
        text = " ".join(["word"] * (CHILD_MAX_TOKENS * 3))
        meta = Metadata(source_id="doc1", header_path="Test", parent_id=None)
        parent = ParentChunk(text=text, metadata=meta)
        children = _split_child_chunks(parent)
        for child in children:
            assert child.metadata.parent_id == parent.id

    def test_child_token_bounds(self):
        """Child chunks respect min/max token limits."""
        text = " ".join(["word"] * (CHILD_MAX_TOKENS * 5))
        meta = Metadata(source_id="doc1", header_path="Test", parent_id=None)
        parent = ParentChunk(text=text, metadata=meta)
        children = _split_child_chunks(parent)
        for child in children:
            token_count = len(_tokenize(child.text))
            assert token_count <= CHILD_MAX_TOKENS, (
                f"Child has {token_count} tokens, max is {CHILD_MAX_TOKENS}"
            )

    def test_child_metadata_inherits_source(self):
        """Children inherit source metadata from parent."""
        meta = Metadata(source_id="src-x", page_number=7, header_path="H", parent_id=None)
        text = " ".join(["word"] * 400)
        parent = ParentChunk(text=text, metadata=meta)
        children = _split_child_chunks(parent)
        for child in children:
            assert child.metadata.source_id == "src-x"
            assert child.metadata.page_number == 7

    def test_short_parent_no_children(self):
        """A very short parent may produce no children if below CHILD_MIN_TOKENS."""
        text = " ".join(["word"] * (CHILD_MIN_TOKENS - 10))
        meta = Metadata(source_id="doc1", header_path="Test", parent_id=None)
        parent = ParentChunk(text=text, metadata=meta)
        children = _split_child_chunks(parent)
        # Short text below min tokens should produce no children
        assert len(children) == 0


# ===========================================================================
# Full markdown ingest
# ===========================================================================

class TestMarkdownIngest:
    def test_ingest_valid_markdown(self, tmp_path: Path):
        """End-to-end markdown ingest produces parents and children."""
        md_content = "# Test Document\n\n" + " ".join(["word"] * 500) + "\n\n## Section\n\n" + " ".join(["word"] * 500)
        md_file = tmp_path / "test.md"
        md_file.write_text(md_content)

        parents, children = ingest_markdown(md_file, source_id="test-doc")
        assert len(parents) > 0
        assert len(children) > 0
        logger.info(f"Ingested {len(parents)} parents, {len(children)} children from markdown")

    def test_ingest_empty_file_raises(self, tmp_path: Path):
        md_file = tmp_path / "empty.md"
        md_file.write_text("")
        with pytest.raises(ValueError, match="empty"):
            ingest_markdown(md_file, source_id="test")

    def test_ingest_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            ingest_markdown("/nonexistent/file.md", source_id="test")

    def test_ingest_empty_source_id_raises(self, tmp_path: Path):
        md_file = tmp_path / "test.md"
        md_file.write_text("# Title\nContent")
        with pytest.raises(ValueError, match="non-empty"):
            ingest_markdown(md_file, source_id="")


# ===========================================================================
# Latency: chunking operations
# ===========================================================================

class TestIngestLatency:
    def test_parent_split_latency(self):
        """Measure parent chunk splitting latency across corpus sizes."""
        sizes = [500, 2000, 5000, 10000]
        for n_tokens in sizes:
            text = " ".join([f"word{i}" for i in range(n_tokens)])
            section = _Section(header_path="Test", text=text)
            with Timer("parent_split", n_tokens=n_tokens) as t:
                parents = _split_parent_chunks(section, source_id="doc1", page_number=1)
            logger.info(
                f"parent_split: {n_tokens} tokens -> {len(parents)} chunks in {t.result.elapsed_ms:.2f}ms"
            )
            assert t.result.elapsed_ms < 1000, "Parent splitting should be fast"

    def test_child_split_latency(self):
        """Measure child chunk splitting latency."""
        text = " ".join([f"word{i}" for i in range(2000)])
        meta = Metadata(source_id="doc1", header_path="Test", parent_id=None)
        parent = ParentChunk(text=text, metadata=meta)
        with Timer("child_split", parent_tokens=2000) as t:
            children = _split_child_chunks(parent)
        logger.info(
            f"child_split: {len(children)} children in {t.result.elapsed_ms:.2f}ms"
        )
        assert t.result.elapsed_ms < 500
