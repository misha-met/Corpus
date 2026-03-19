"""Tests for data layer: Metadata, ParentChunk, ChildChunk models and ingest chunking."""
from __future__ import annotations

import re
import tempfile
import uuid
from pathlib import Path

import pytest
from unittest.mock import patch

from src.models import ChildChunk, Metadata, ParentChunk
from src.ingest import (
    CHILD_MAX_TOKENS,
    CHILD_MIN_TOKENS,
    CHILD_OVERLAP_SENTENCES,
    CHILD_TARGET_TOKENS,
    PARENT_MAX_TOKENS,
    PARENT_MIN_TOKENS,
    PARENT_OVERLAP_TOKENS,
    PARENT_TARGET_TOKENS,
    _parse_markdown_sections,
    _split_child_chunks,
    _split_long_sentence_on_clause,
    _split_sentences,
    _split_parent_chunks,
    _split_tokens,
    _tokenize,
    _detokenize,
    _Section,
    clean_ocr_artifacts,
    ingest_markdown,
    _sample_context,
    _format_page_marker,
    _parse_page_markers,
    _marker_page_range,
    _PDF_STRATEGIES,
    _extract_pdfminer,
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
            start_page=5,
            end_page=6,
            page_label="v",
            display_page="v",
            header_path="Ch1 > Sec2",
            parent_id="pid-123",
        )
        assert m.page_number == 5
        assert m.start_page == 5
        assert m.end_page == 6
        assert m.display_page == "v"
        assert m.parent_id == "pid-123"

    def test_invalid_page_range_rejected(self):
        with pytest.raises(Exception):
            Metadata(
                source_id="doc1",
                page_number=5,
                start_page=9,
                end_page=7,
                header_path="Doc",
            )

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


class TestPageMarkers:
    def test_format_page_marker(self):
        assert _format_page_marker(7) == "[Page 7]"

    def test_parse_page_markers(self):
        text = "[Page 3] intro [Page 4] body [PAGE 5] tail"
        assert _parse_page_markers(text) == [3, 4, 5]

    def test_marker_page_range(self):
        assert _marker_page_range("No markers here") == (None, None)
        assert _marker_page_range("[Page 10] text [Page 12]") == (10, 12)


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

    def test_parent_chunk_derives_range_from_marker(self):
        section = _Section(
            header_path="Test",
            text="[Page 11] " + " ".join(["word"] * 250),
        )
        parents = _split_parent_chunks(
            section,
            source_id="doc1",
            page_number=3,
            display_page="iii",
        )
        assert len(parents) == 1
        parent = parents[0]
        assert parent.metadata.start_page == 11
        assert parent.metadata.end_page == 11
        assert parent.metadata.page_number == 11
        # Keep legacy display-page semantics when provided by extractor.
        assert parent.metadata.display_page == "iii"


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
        """Child chunks stay near target size with sentence-aware soft limits."""
        sentence = " ".join(["word"] * 25) + "."
        text = " ".join([sentence] * 120)
        meta = Metadata(source_id="doc1", header_path="Test", parent_id=None)
        parent = ParentChunk(text=text, metadata=meta)
        children = _split_child_chunks(parent)
        assert len(children) > 0
        avg_tokens = sum(len(_tokenize(child.text)) for child in children) / len(children)
        assert avg_tokens == pytest.approx(CHILD_TARGET_TOKENS, rel=0.15)

        for child in children:
            token_count = len(_tokenize(child.text))
            assert token_count >= CHILD_MIN_TOKENS
            assert token_count <= int(CHILD_TARGET_TOKENS * 1.5), (
                f"Child has {token_count} tokens, expected soft cap near target"
            )

    def test_child_chunks_preserve_sentence_boundaries(self):
        text = " ".join([f"Sentence {i} ends here." for i in range(1, 80)])
        meta = Metadata(source_id="doc1", header_path="Test", parent_id=None)
        parent = ParentChunk(text=text, metadata=meta)
        children = _split_child_chunks(parent)
        assert len(children) > 0
        for child in children:
            stripped = child.text.strip()
            assert stripped[-1] in {".", "!", "?"}

    def test_child_chunk_sentence_overlap(self):
        sentence_tokens = " ".join(["word"] * 24)
        text = " ".join([f"S{i} {sentence_tokens}." for i in range(1, 80)])
        meta = Metadata(source_id="doc1", header_path="Test", parent_id=None)
        parent = ParentChunk(text=text, metadata=meta)
        children = _split_child_chunks(parent)
        if len(children) > 1:
            prev_sentences = _split_sentences(children[0].text)
            next_sentences = _split_sentences(children[1].text)
            assert next_sentences[:CHILD_OVERLAP_SENTENCES] == prev_sentences[-CHILD_OVERLAP_SENTENCES:]

    def test_long_sentence_clause_fallback(self):
        lead = " ".join(["intro"] * 120)
        middle = " ".join(["detail"] * 90)
        tail = " ".join(["evidence"] * 40)
        long_sentence = (
            f"{lead}, and {middle}, which {tail}."
        )
        parts = _split_long_sentence_on_clause(long_sentence, CHILD_TARGET_TOKENS)
        assert len(parts) >= 2
        assert all(part.strip() for part in parts)
        assert "," in parts[0] or ";" in parts[0]

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
        """A very short parent produces a sole child (rather than losing data)."""
        text = " ".join(["word"] * (CHILD_MIN_TOKENS - 10))
        meta = Metadata(source_id="doc1", header_path="Test", parent_id=None)
        parent = ParentChunk(text=text, metadata=meta)
        children = _split_child_chunks(parent)
        # Short text below min tokens should produce a sole child chunk
        assert len(children) == 1
        assert children[0].text == text

    def test_child_chunk_derives_page_range_from_marker(self):
        text = "[Page 12] " + " ".join(["word"] * (CHILD_MIN_TOKENS + 20))
        meta = Metadata(
            source_id="doc1",
            page_number=2,
            start_page=2,
            end_page=2,
            header_path="Test",
            parent_id=None,
        )
        parent = ParentChunk(text=text, metadata=meta)
        children = _split_child_chunks(parent)
        assert children
        child = children[0]
        assert child.metadata.start_page == 12
        assert child.metadata.end_page == 12
        assert child.metadata.page_number == 12
        assert child.metadata.display_page == "12"

    def test_child_chunk_falls_back_to_parent_range_when_marker_missing(self):
        text = " ".join(["word"] * (CHILD_MIN_TOKENS + 20))
        meta = Metadata(
            source_id="doc1",
            page_number=6,
            start_page=6,
            end_page=7,
            display_page="6",
            header_path="Test",
            parent_id=None,
        )
        parent = ParentChunk(text=text, metadata=meta)
        children = _split_child_chunks(parent)
        assert children
        for child in children:
            assert child.metadata.start_page == 6
            assert child.metadata.end_page == 7
            assert child.metadata.page_number == 6


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


class TestPdfStrategyOrder:
    def test_pdf_strategy_order_quality_first(self):
        names = [strategy.__name__ for strategy in _PDF_STRATEGIES]
        assert names == [
            "_extract_pymupdf",
            "_extract_pypdf",
            "_extract_pdfminer",
            "_extract_ocr",
        ]

    def test_pdfminer_extracts_per_page_via_page_numbers(self, tmp_path: Path):
        calls: list[list[int]] = []

        class _Reader:
            pages = [object(), object(), object()]

        def _fake_extract_text(_path: str, page_numbers=None):
            calls.append(list(page_numbers or []))
            index = calls[-1][0]
            return f"page {index + 1}"

        with patch("pdfminer.high_level.extract_text", side_effect=_fake_extract_text):
            pages = _extract_pdfminer(tmp_path / "doc.pdf", reader=_Reader(), page_offset=1)

        assert calls == [[0], [1], [2]]
        assert len(pages) == 3


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


# ===========================================================================
# _sample_context
# ===========================================================================

class TestSampleContext:
    def test_short_text_unchanged(self):
        text = "hello world"
        assert _sample_context(text, 12_000) == text

    def test_exact_limit_unchanged(self):
        text = "x" * 12_000
        assert _sample_context(text, 12_000) == text

    def test_long_text_truncated_to_limit(self):
        text = "a" * 30_000
        result = _sample_context(text, 12_000)
        # Three thirds of 4000 chars each + two "\n\n[...]\n\n" separators (10 chars each) = 12020
        assert len(result) <= 12_100

    def test_long_text_contains_separator(self):
        text = "a" * 30_000
        result = _sample_context(text, 12_000)
        assert "[...]" in result

    def test_long_text_starts_with_head(self):
        text = "START" + "a" * 29_995
        result = _sample_context(text, 12_000)
        assert result.startswith("START")

    def test_long_text_ends_with_tail(self):
        text = "a" * 29_995 + "FINISH"
        result = _sample_context(text, 12_000)
        assert result.endswith("FINISH")
