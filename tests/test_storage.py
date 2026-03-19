"""Tests for unified LanceDB storage engine."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import lancedb
import pyarrow as pa
import pytest

from src.models import ChildChunk, Metadata, ParentChunk
from src.storage import StorageConfig, StorageEngine
from tests.conftest import (
    MockEmbeddingModel,
    Timer,
    generate_parent_child_corpus,
    get_test_logger,
)

logger = get_test_logger("storage")


# ===========================================================================
# Parent store (LanceDB)
# ===========================================================================

class TestParentStore:
    def test_add_and_retrieve_parent(self, tmp_storage: StorageEngine):
        """Stored parent text should be retrievable by parent_id."""
        sources = tmp_storage.list_source_ids()
        assert len(sources) > 0, "Should have source_ids after ingest"
        texts = tmp_storage.get_parent_texts_by_source(source_id=sources[0])
        assert len(texts) > 0, "Should have parent texts for first source"

    def test_missing_parent_returns_none(self, tmp_storage: StorageEngine):
        assert tmp_storage.get_parent_text("nonexistent-id") is None

    def test_list_source_ids(self, tmp_storage: StorageEngine):
        sources = tmp_storage.list_source_ids()
        assert "test_doc_linguistics" in sources
        assert "test_doc_philosophy" in sources

    def test_get_parent_texts_by_source(self, tmp_storage: StorageEngine):
        texts = tmp_storage.get_parent_texts_by_source(source_id="test_doc_linguistics")
        assert len(texts) > 0

    def test_get_parent_text_handles_quotes_in_id(self, tmp_path: Path):
        config = StorageConfig(lance_dir=tmp_path / "lance")
        engine = StorageEngine(config)
        quoted_parent = ParentChunk(
            id="parent-'quoted'",
            text="Quoted parent text",
            metadata=Metadata(
                source_id="doc-1",
                page_number=1,
                page_label="1",
                display_page="1",
                header_path="Document",
                parent_id=None,
            ),
        )
        engine.add_parents([quoted_parent])
        assert engine.get_parent_text("parent-'quoted'") == "Quoted parent text"

    def test_get_parent_texts_batch(self, tmp_storage: StorageEngine, mock_embedder: MockEmbeddingModel):
        q_vec = mock_embedder.encode(["Chomsky theory epistemology"], normalize_embeddings=True)[0]
        hits = tmp_storage.hybrid_search(query_text="Chomsky theory epistemology", query_vector=q_vec, top_k=5)
        parent_ids = [r["metadata"].get("parent_id") for r in hits if r.get("metadata")]
        parent_ids = [pid for pid in parent_ids if isinstance(pid, str)]
        assert parent_ids

        found = tmp_storage.get_parent_texts(parent_ids + ["nonexistent-id"])
        for parent_id in parent_ids:
            assert parent_id in found
        assert "nonexistent-id" not in found


# ===========================================================================
# Child chunk store (LanceDB vectors + FTS)
# ===========================================================================

class TestChildStore:
    def test_hybrid_search_returns_results(self, tmp_storage: StorageEngine, mock_embedder: MockEmbeddingModel):
        """Hybrid search should return child chunks."""
        q_vec = mock_embedder.encode(["Chomsky grammar"], normalize_embeddings=True)[0]
        results = tmp_storage.hybrid_search(
            query_text="Chomsky grammar", query_vector=q_vec, top_k=5,
        )
        assert len(results) > 0

    def test_hybrid_search_top_k_respected(self, tmp_storage: StorageEngine, mock_embedder: MockEmbeddingModel):
        """Returned results should not exceed top_k."""
        q_vec = mock_embedder.encode(["test query"], normalize_embeddings=True)[0]
        results = tmp_storage.hybrid_search(
            query_text="test query", query_vector=q_vec, top_k=2,
        )
        assert len(results) <= 2

    def test_hybrid_search_returns_metadata(self, tmp_storage: StorageEngine, mock_embedder: MockEmbeddingModel):
        """Each result should include metadata with source_id."""
        q_vec = mock_embedder.encode(["Chomsky grammar"], normalize_embeddings=True)[0]
        results = tmp_storage.hybrid_search(
            query_text="Chomsky grammar", query_vector=q_vec, top_k=3,
        )
        for r in results:
            assert "source_id" in r["metadata"]

    def test_get_children_by_ids(self, tmp_storage: StorageEngine, mock_embedder: MockEmbeddingModel):
        """get_children_by_ids should return text and metadata."""
        q_vec = mock_embedder.encode(["test"], normalize_embeddings=True)[0]
        search_results = tmp_storage.hybrid_search(
            query_text="test", query_vector=q_vec, top_k=3,
        )
        ids = [r["id"] for r in search_results]
        fetched = tmp_storage.get_children_by_ids(ids)
        assert len(fetched) == len(ids)
        for cid, data in fetched.items():
            assert "text" in data
            assert "metadata" in data

    def test_empty_ids_returns_empty(self, tmp_storage: StorageEngine):
        assert tmp_storage.get_children_by_ids([]) == {}

    def test_hybrid_search_deterministic(self, tmp_storage: StorageEngine, mock_embedder: MockEmbeddingModel):
        """Same query should produce same results on repeated calls."""
        q_vec = mock_embedder.encode(["Chomsky language"], normalize_embeddings=True)[0]
        r1 = tmp_storage.hybrid_search(
            query_text="Chomsky language", query_vector=q_vec, top_k=5,
        )
        r2 = tmp_storage.hybrid_search(
            query_text="Chomsky language", query_vector=q_vec, top_k=5,
        )
        assert [r["id"] for r in r1] == [r["id"] for r in r2]

    def test_hybrid_search_source_filter(self, tmp_storage: StorageEngine, mock_embedder: MockEmbeddingModel):
        """Source filter should restrict results to one source."""
        q_vec = mock_embedder.encode(["knowledge epistemology"], normalize_embeddings=True)[0]
        results = tmp_storage.hybrid_search(
            query_text="knowledge epistemology", query_vector=q_vec,
            top_k=10, source_id="test_doc_philosophy",
        )
        for r in results:
            assert r["metadata"].get("source_id") == "test_doc_philosophy"

    def test_hybrid_search_source_filter_handles_quotes(self, tmp_path: Path, mock_embedder: MockEmbeddingModel):
        config = StorageConfig(lance_dir=tmp_path / "lance")
        engine = StorageEngine(config)
        quoted_source = "doc-'quoted'"
        parent = ParentChunk(
            id="p-quoted",
            text="Parent text",
            metadata=Metadata(
                source_id=quoted_source,
                page_number=1,
                page_label="1",
                display_page="1",
                header_path="Document",
                parent_id=None,
            ),
        )
        child = ChildChunk(
            text="quoted source child text",
            metadata=Metadata(
                source_id=quoted_source,
                page_number=1,
                page_label="1",
                display_page="1",
                header_path="Document",
                parent_id=parent.id,
            ),
        )
        engine.add_parents([parent])
        embedding = mock_embedder.encode([child.text], normalize_embeddings=True)
        engine.add_children([child], embeddings=embedding)

        q_vec = mock_embedder.encode(["quoted source child"], normalize_embeddings=True)[0]
        results = engine.hybrid_search(
            query_text="quoted source child",
            query_vector=q_vec,
            top_k=5,
            source_id=quoted_source,
        )
        assert results
        assert all(r["metadata"].get("source_id") == quoted_source for r in results)

    def test_persist_reopen(self, tmp_path: Path, mock_embedder: MockEmbeddingModel):
        """Data should survive close + reopen."""
        config = StorageConfig(lance_dir=tmp_path / "lance")
        engine = StorageEngine(config)
        parents, children = generate_parent_child_corpus()
        engine.add_parents(parents)
        texts = [c.text for c in children]
        embeddings = mock_embedder.encode(texts, normalize_embeddings=True)
        engine.add_children(children, embeddings=embeddings)
        engine.close()

        engine2 = StorageEngine(config)
        sources = engine2.list_source_ids()
        assert len(sources) > 0, "Data should persist across sessions"
        engine2.close()

    def test_range_metadata_round_trip(self, tmp_path: Path, mock_embedder: MockEmbeddingModel):
        config = StorageConfig(lance_dir=tmp_path / "lance")
        engine = StorageEngine(config)

        parent = ParentChunk(
            id="p-range",
            text="[Page 10] Parent context",
            metadata=Metadata(
                source_id="doc-range",
                page_number=10,
                start_page=10,
                end_page=12,
                page_label="10",
                display_page="10",
                header_path="Document",
                parent_id=None,
            ),
        )
        child = ChildChunk(
            id="c-range",
            text="[Page 10] child chunk",
            metadata=Metadata(
                source_id="doc-range",
                page_number=10,
                start_page=10,
                end_page=11,
                page_label="10",
                display_page="10",
                header_path="Document",
                parent_id=parent.id,
            ),
        )

        engine.add_parents([parent])
        embedding = mock_embedder.encode([child.text], normalize_embeddings=True)
        engine.add_children([child], embeddings=embedding)

        fetched = engine.get_children_by_ids([child.id])
        assert child.id in fetched
        meta = fetched[child.id]["metadata"]
        assert meta["start_page"] == 10
        assert meta["end_page"] == 11

        engine.close()

    def test_existing_tables_migrate_range_columns(self, tmp_path: Path):
        lance_dir = tmp_path / "lance"
        db = lancedb.connect(str(lance_dir))

        db.create_table(
            "child_chunks",
            [
                {
                    "id": "c1",
                    "text": "legacy child",
                    "source_id": "doc-legacy",
                    "page_number": 1,
                    "page_label": "1",
                    "display_page": "1",
                    "header_path": "Document",
                    "parent_id": "p1",
                    "vector": [0.1, 0.2, 0.3],
                }
            ],
        )
        db.create_table(
            "parent_chunks",
            [
                {
                    "parent_id": "p1",
                    "source_id": "doc-legacy",
                    "page_number": 1,
                    "page_label": "1",
                    "display_page": "1",
                    "header_path": "Document",
                    "text": "legacy parent",
                }
            ],
        )

        engine = StorageEngine(StorageConfig(lance_dir=lance_dir))
        assert engine._table is not None
        assert engine._parents is not None
        assert "start_page" in engine._table.schema.names
        assert "end_page" in engine._table.schema.names
        assert "start_page" in engine._parents.schema.names
        assert "end_page" in engine._parents.schema.names
        engine.close()


# ===========================================================================
# Source summaries (LanceDB)
# ===========================================================================

class TestSourceSummaries:
    def test_upsert_and_get_summary(self, tmp_storage: StorageEngine):
        tmp_storage.upsert_source_summary(source_id="test_doc", summary="A test summary.")
        summaries = tmp_storage.get_source_summaries()
        assert "test_doc" in summaries
        assert summaries["test_doc"] == "A test summary."

    def test_upsert_overwrites(self, tmp_storage: StorageEngine):
        tmp_storage.upsert_source_summary(source_id="doc_x", summary="v1")
        tmp_storage.upsert_source_summary(source_id="doc_x", summary="v2")
        summaries = tmp_storage.get_source_summaries()
        assert summaries["doc_x"] == "v2"

    def test_empty_source_id_rejected(self, tmp_storage: StorageEngine):
        with pytest.raises(ValueError):
            tmp_storage.upsert_source_summary(source_id="", summary="text")

    def test_empty_summary_rejected(self, tmp_storage: StorageEngine):
        with pytest.raises(ValueError):
            tmp_storage.upsert_source_summary(source_id="doc", summary="")

    def test_upsert_summary_handles_quotes_in_source_id(self, tmp_storage: StorageEngine):
        quoted_source = "doc-'quoted'"
        tmp_storage.upsert_source_summary(source_id=quoted_source, summary="v1")
        tmp_storage.upsert_source_summary(source_id=quoted_source, summary="v2")
        summaries = tmp_storage.get_source_summaries()
        assert summaries[quoted_source] == "v2"

    def test_upsert_and_read_citation_reference(self, tmp_storage: StorageEngine):
        tmp_storage.upsert_source_summary(
            source_id="doc_citation",
            summary="summary",
            citation_reference="Smith (2024) Test",
        )
        detail = tmp_storage.get_source_detail("doc_citation")
        assert detail is not None
        assert detail["citation_reference"] == "Smith (2024) Test"

    def test_persist_source_page_offset_updates_metadata_without_summary(self, tmp_storage: StorageEngine):
        tmp_storage.persist_source_page_offset(
            source_id="doc_meta",
            page_offset=3,
            source_path="/tmp/doc.pdf",
            snapshot_path="/tmp/doc.snapshot.txt",
            citation_reference="Doe (2025) Metadata",
        )
        detail = tmp_storage.get_source_detail("doc_meta")
        assert detail is not None
        assert detail["summary"] == ""
        assert detail["source_path"] == "/tmp/doc.pdf"
        assert detail["snapshot_path"] == "/tmp/doc.snapshot.txt"
        assert detail["page_offset"] == 3
        assert detail["citation_reference"] == "Doe (2025) Metadata"


class TestFtsPolicy:
    def test_storage_config_defaults_to_immediate_policy(self):
        cfg = StorageConfig(lance_dir=Path(tempfile.mkdtemp()))
        assert cfg.fts_rebuild_policy == "immediate"

    def test_get_fts_status_includes_policy_dirty_and_pending_rows(self, tmp_storage: StorageEngine):
        status = tmp_storage.get_fts_status()
        assert status["fts_policy"] in {"immediate", "deferred", "batch"}
        assert isinstance(status["fts_dirty"], bool)
        assert isinstance(status["fts_pending_rows"], int)


# ===========================================================================
# Latency: index operations
# ===========================================================================

class TestStorageLatency:
    def test_hybrid_search_latency(self, tmp_storage: StorageEngine, mock_embedder: MockEmbeddingModel):
        """Measure hybrid search latency."""
        q_vec = mock_embedder.encode(["Chomsky theory"], normalize_embeddings=True)[0]
        top_k_values = [5, 10, 20]
        for k in top_k_values:
            with Timer("hybrid_search", top_k=k) as t:
                tmp_storage.hybrid_search(
                    query_text="Chomsky theory", query_vector=q_vec, top_k=k,
                )
            logger.info(f"hybrid_search top_k={k}: {t.result.elapsed_ms:.2f}ms")

    def test_parent_lookup_latency(self, tmp_storage: StorageEngine):
        """Measure parent text lookup latency."""
        parents, _ = generate_parent_child_corpus()
        for parent in parents[:3]:
            with Timer("parent_lookup", parent_id=parent.id) as t:
                tmp_storage.get_parent_text(parent.id)
            logger.info(f"parent_lookup: {t.result.elapsed_ms:.2f}ms")


# ===========================================================================
# Person mentions table
# ===========================================================================


class TestPersonMentionsStorage:
    @staticmethod
    def _sample_rows() -> list[dict[str, object]]:
        return [
            {
                "id": "pm-1",
                "source_id": "doc_a",
                "chunk_id": "c-a1",
                "raw_name": "Noam Chomsky",
                "canonical_name": "Noam Chomsky",
                "confidence": 0.95,
                "method": "new",
                "role_hint": "author",
                "context_snippet": "written by Noam Chomsky",
            },
            {
                "id": "pm-2",
                "source_id": "doc_b",
                "chunk_id": "c-b1",
                "raw_name": "Chomsky",
                "canonical_name": "Noam Chomsky",
                "confidence": 0.88,
                "method": "fuzzy_last",
                "role_hint": "cited",
                "context_snippet": "according to Chomsky",
            },
            {
                "id": "pm-3",
                "source_id": "doc_b",
                "chunk_id": "c-b2",
                "raw_name": "Michel Foucault",
                "canonical_name": "Michel Foucault",
                "confidence": 0.62,
                "method": "new",
                "role_hint": "subject",
                "context_snippet": "focuses on Michel Foucault",
            },
        ]

    def test_person_mentions_table_auto_create(self, tmp_path: Path) -> None:
        engine = StorageEngine(StorageConfig(lance_dir=tmp_path / "lance-auto"))
        assert engine.get_person_mentions(min_confidence=0.0) == []

        engine.upsert_person_mentions(self._sample_rows()[:1])
        rows = engine.get_person_mentions(min_confidence=0.0)
        assert len(rows) == 1
        assert rows[0]["canonical_name"] == "Noam Chomsky"
        engine.close()

    def test_person_mentions_confidence_schema_float32(self) -> None:
        schema = StorageEngine._PERSON_MENTIONS_SCHEMA
        confidence_field = schema.field("confidence")
        assert confidence_field.type == pa.float32()

    def test_person_mentions_upsert_get_delete(self, tmp_storage: StorageEngine) -> None:
        tmp_storage.upsert_person_mentions(self._sample_rows())

        all_rows = tmp_storage.get_person_mentions(min_confidence=0.0)
        assert len(all_rows) == 3

        canonical_rows = tmp_storage.get_person_mentions_by_canonical(
            "Noam Chomsky",
            min_confidence=0.0,
        )
        assert len(canonical_rows) == 2

        row = tmp_storage.get_person_mention("pm-2")
        assert row is not None
        assert row["raw_name"] == "Chomsky"

        tmp_storage.delete_person_mention("pm-2")
        assert tmp_storage.get_person_mention("pm-2") is None

    def test_person_mentions_filters(self, tmp_storage: StorageEngine) -> None:
        tmp_storage.upsert_person_mentions(self._sample_rows())

        doc_a = tmp_storage.get_person_mentions(
            source_ids=["doc_a"],
            min_confidence=0.0,
        )
        assert len(doc_a) == 1
        assert all(row["source_id"] == "doc_a" for row in doc_a)

        high_conf = tmp_storage.get_person_mentions(min_confidence=0.9)
        assert len(high_conf) == 1
        assert high_conf[0]["id"] == "pm-1"

    def test_recreate_person_mentions_table(self, tmp_storage: StorageEngine) -> None:
        tmp_storage.upsert_person_mentions(self._sample_rows()[:2])
        assert len(tmp_storage.get_person_mentions(min_confidence=0.0)) == 2

        table = tmp_storage._recreate_person_mentions_table()
        assert table is not None
        assert tmp_storage.get_person_mentions(min_confidence=0.0) == []

    def test_merge_person_canonical_names_rewrites_rows(self, tmp_storage: StorageEngine) -> None:
        tmp_storage.upsert_person_mentions(self._sample_rows())

        merged = tmp_storage.merge_person_canonical_names(
            "Michel Foucault",
            "Noam Chomsky",
        )
        assert merged == 1

        old_rows = tmp_storage.get_person_mentions_by_canonical(
            "Michel Foucault",
            min_confidence=0.0,
        )
        assert old_rows == []

        merged_rows = tmp_storage.get_person_mentions_by_canonical(
            "Noam Chomsky",
            min_confidence=0.0,
        )
        assert len(merged_rows) == 3
        assert all(row["canonical_name"] == "Noam Chomsky" for row in merged_rows)

    def test_delete_source_cascades_person_mentions(self, tmp_storage: StorageEngine) -> None:
        parent = ParentChunk(
            id="p-del",
            text="delete-source test",
            metadata=Metadata(
                source_id="doc_del",
                page_number=1,
                page_label="1",
                display_page="1",
                header_path="Document",
                parent_id=None,
            ),
        )
        tmp_storage.add_parents([parent])
        tmp_storage.upsert_person_mentions(
            [
                {
                    "id": "pm-del",
                    "source_id": "doc_del",
                    "chunk_id": "c-del",
                    "raw_name": "Noam Chomsky",
                    "canonical_name": "Noam Chomsky",
                    "confidence": 0.9,
                    "method": "new",
                    "role_hint": "author",
                    "context_snippet": "context",
                },
                {
                    "id": "pm-keep",
                    "source_id": "doc_keep",
                    "chunk_id": "c-keep",
                    "raw_name": "Michel Foucault",
                    "canonical_name": "Michel Foucault",
                    "confidence": 0.9,
                    "method": "new",
                    "role_hint": "subject",
                    "context_snippet": "context",
                },
            ]
        )

        tmp_storage.delete_source("doc_del")

        deleted_rows = tmp_storage.get_person_mentions(source_id="doc_del", min_confidence=0.0)
        kept_rows = tmp_storage.get_person_mentions(source_id="doc_keep", min_confidence=0.0)
        assert deleted_rows == []
        assert len(kept_rows) == 1

    def test_delete_source_cascades_geo_mentions(self, tmp_storage: StorageEngine) -> None:
        parent = ParentChunk(
            id="p-geo-del",
            text="delete-source geo test",
            metadata=Metadata(
                source_id="doc_geo_del",
                page_number=1,
                page_label="1",
                display_page="1",
                header_path="Document",
                parent_id=None,
            ),
        )
        tmp_storage.add_parents([parent])
        tmp_storage.upsert_geo_mentions(
            [
                {
                    "id": "gm-del",
                    "source_id": "doc_geo_del",
                    "chunk_id": "c-geo-del",
                    "place_name": "Paris",
                    "matched_input": "Paris",
                    "matched_on": "paris",
                    "geonameid": 2988507,
                    "lat": 48.8566,
                    "lon": 2.3522,
                    "confidence": 0.95,
                    "method": "exact",
                },
                {
                    "id": "gm-keep",
                    "source_id": "doc_geo_keep",
                    "chunk_id": "c-geo-keep",
                    "place_name": "London",
                    "matched_input": "London",
                    "matched_on": "london",
                    "geonameid": 2643743,
                    "lat": 51.5072,
                    "lon": -0.1276,
                    "confidence": 0.94,
                    "method": "exact",
                },
            ]
        )

        tmp_storage.delete_source("doc_geo_del")

        deleted_rows = tmp_storage.get_geo_mentions(source_id="doc_geo_del", min_confidence=0.0)
        kept_rows = tmp_storage.get_geo_mentions(source_id="doc_geo_keep", min_confidence=0.0)
        assert deleted_rows == []
        assert len(kept_rows) == 1

    def test_delete_source_cascades_geo_and_person_mentions(self, tmp_storage: StorageEngine) -> None:
        parent = ParentChunk(
            id="p-both-del",
            text="delete-source combined entity test",
            metadata=Metadata(
                source_id="doc_both_del",
                page_number=1,
                page_label="1",
                display_page="1",
                header_path="Document",
                parent_id=None,
            ),
        )
        tmp_storage.add_parents([parent])

        tmp_storage.upsert_geo_mentions(
            [
                {
                    "id": "gm-both-del",
                    "source_id": "doc_both_del",
                    "chunk_id": "c-both-del",
                    "place_name": "Rome",
                    "matched_input": "Rome",
                    "matched_on": "rome",
                    "geonameid": 3169070,
                    "lat": 41.9028,
                    "lon": 12.4964,
                    "confidence": 0.96,
                    "method": "exact",
                },
                {
                    "id": "gm-both-keep",
                    "source_id": "doc_both_keep",
                    "chunk_id": "c-both-keep",
                    "place_name": "Athens",
                    "matched_input": "Athens",
                    "matched_on": "athens",
                    "geonameid": 264371,
                    "lat": 37.9838,
                    "lon": 23.7275,
                    "confidence": 0.93,
                    "method": "exact",
                },
            ]
        )
        tmp_storage.upsert_person_mentions(
            [
                {
                    "id": "pm-both-del",
                    "source_id": "doc_both_del",
                    "chunk_id": "c-both-del",
                    "raw_name": "Noam Chomsky",
                    "canonical_name": "Noam Chomsky",
                    "confidence": 0.92,
                    "method": "new",
                    "role_hint": "author",
                    "context_snippet": "context",
                },
                {
                    "id": "pm-both-keep",
                    "source_id": "doc_both_keep",
                    "chunk_id": "c-both-keep",
                    "raw_name": "Michel Foucault",
                    "canonical_name": "Michel Foucault",
                    "confidence": 0.9,
                    "method": "new",
                    "role_hint": "subject",
                    "context_snippet": "context",
                },
            ]
        )

        tmp_storage.delete_source("doc_both_del")

        assert tmp_storage.get_geo_mentions(source_id="doc_both_del", min_confidence=0.0) == []
        assert tmp_storage.get_person_mentions(source_id="doc_both_del", min_confidence=0.0) == []
        assert len(tmp_storage.get_geo_mentions(source_id="doc_both_keep", min_confidence=0.0)) == 1
        assert len(tmp_storage.get_person_mentions(source_id="doc_both_keep", min_confidence=0.0)) == 1
