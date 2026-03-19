"""Tests for source management API endpoints: list, ingest, delete, content.

Uses a mock engine with real StorageEngine (tmpdir-based LanceDB) and
real source_cache for full integration testing of the endpoints.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import patch

import httpx
import pytest

from src.api import app
from src.source_cache import save_snapshot
from src.storage import StorageConfig, StorageEngine


# ---------------------------------------------------------------------------
# Mock engine with real storage
# ---------------------------------------------------------------------------


class MockEngineWithStorage:
    """Mock engine that wraps a real StorageEngine for endpoint testing.

    Does NOT load ML models — ingest is simulated by directly adding
    data to storage.
    """

    def __init__(self, storage: StorageEngine):
        self._storage = storage
        self.simulate_ner_unavailable = False

    @property
    def storage(self) -> StorageEngine:
        return self._storage

    def ingest(
        self,
        file_path,
        *,
        source_id,
        summarize=True,
        page_number=None,
        geotag=False,
        peopletag=False,
        citation_reference=None,
        page_offset=1,
    ):
        """Simulate ingest by storing parent chunks and a summary."""
        from src.models import Metadata, ParentChunk
        _ = page_number, page_offset, citation_reference

        # Read the file content
        text = Path(file_path).read_text(encoding="utf-8")

        # Create a parent chunk
        parent = ParentChunk(
            text=text,
            metadata=Metadata(
                source_id=source_id,
                page_number=1,
                header_path="Document",
            ),
        )
        self._storage.add_parents([parent])

        # Create a summary
        if summarize:
            self._storage.upsert_source_summary(
                source_id=source_id,
                summary=f"Summary of {source_id}",
                source_path=str(Path(file_path).resolve()),
            )

        geotag_diag = None
        peopletag_diag = None

        if geotag:
            if self.simulate_ner_unavailable:
                geotag_diag = SimpleNamespace(
                    ner_available=False,
                    method="regex_fallback",
                    warning="GLiNER unavailable; regex fallback used",
                )
            else:
                geotag_diag = SimpleNamespace(
                    ner_available=True,
                    method="gliner",
                    warning=None,
                )

        if peopletag:
            if self.simulate_ner_unavailable:
                peopletag_diag = SimpleNamespace(
                    ner_available=False,
                    method="empty",
                    warning="GLiNER unavailable; person extraction returned empty",
                )
            else:
                peopletag_diag = SimpleNamespace(
                    ner_available=True,
                    method="gliner",
                    warning=None,
                )

        return SimpleNamespace(
            source_id=source_id,
            parents_count=1,
            children_count=0,
            summarized=summarize,
            geotag_ner=geotag_diag,
            peopletag_ner=peopletag_diag,
        )

    def query_events(self, *args, **kwargs):
        yield from []

    def close(self):
        self._storage.close()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_storage(tmp_path: Path) -> StorageEngine:
    config = StorageConfig(
        lance_dir=tmp_path / "lance",
        lance_table="test_chunks",
    )
    engine = StorageEngine(config)
    yield engine
    engine.close()


@pytest.fixture
def mock_engine(tmp_storage: StorageEngine) -> MockEngineWithStorage:
    return MockEngineWithStorage(tmp_storage)


@pytest.fixture
def sample_file(tmp_path: Path) -> Path:
    """Create a sample markdown file for testing."""
    f = tmp_path / "sample.md"
    f.write_text("# Test Document\n\nThis is a sample document for testing.", encoding="utf-8")
    return f


# ---------------------------------------------------------------------------
# GET /api/sources — list
# ---------------------------------------------------------------------------


class TestListSources:
    @pytest.mark.anyio
    async def test_empty_list(self, mock_engine) -> None:
        with patch("src.api._get_engine", return_value=mock_engine):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get("/api/sources")
        assert resp.status_code == 200
        data = resp.json()
        assert data["sources"] == []

    @pytest.mark.anyio
    async def test_list_after_ingest(self, mock_engine, sample_file) -> None:
        # Simulate ingest
        mock_engine.ingest(str(sample_file), source_id="test_doc")

        with patch("src.api._get_engine", return_value=mock_engine):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get("/api/sources")

        assert resp.status_code == 200
        sources = resp.json()["sources"]
        assert len(sources) >= 1
        assert any(s["source_id"] == "test_doc" for s in sources)

    @pytest.mark.anyio
    async def test_list_includes_summary(self, mock_engine, sample_file) -> None:
        mock_engine.ingest(str(sample_file), source_id="doc_a")

        with patch("src.api._get_engine", return_value=mock_engine):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get("/api/sources")

        sources = resp.json()["sources"]
        doc = next(s for s in sources if s["source_id"] == "doc_a")
        assert doc["summary"] is not None
        assert "Summary of doc_a" in doc["summary"]


# ---------------------------------------------------------------------------
# GET /api/geo/mentions — source_ids contract
# ---------------------------------------------------------------------------


class TestGeoMentionsContract:
    @staticmethod
    def _seed_mentions(storage: StorageEngine) -> None:
        storage.upsert_geo_mentions(
            [
                {
                    "id": "m-doc-a",
                    "source_id": "doc_a",
                    "chunk_id": "c-doc-a",
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
                    "id": "m-doc-b",
                    "source_id": "doc_b",
                    "chunk_id": "c-doc-b",
                    "place_name": "London",
                    "matched_input": "London",
                    "matched_on": "london",
                    "geonameid": 2643743,
                    "lat": 51.5072,
                    "lon": -0.1276,
                    "confidence": 0.93,
                    "method": "exact",
                },
            ]
        )

    @pytest.mark.anyio
    async def test_source_id_and_source_ids_use_union(self, mock_engine) -> None:
        self._seed_mentions(mock_engine.storage)
        with patch("src.api.app_config.USE_SOURCE_IDS_FILTER", True):
            with patch("src.api._get_engine", return_value=mock_engine):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    resp = await client.get(
                        "/api/geo/mentions",
                        params={
                            "source_id": "doc_a",
                            "source_ids": ["doc_b"],
                            "min_confidence": 0.0,
                        },
                    )

        assert resp.status_code == 200
        mentions = resp.json()["mentions"]
        all_sources = {sid for group in mentions for sid in group.get("source_ids", [])}
        assert "doc_a" in all_sources
        assert "doc_b" in all_sources

    @pytest.mark.anyio
    async def test_empty_source_ids_returns_empty_when_source_id_absent(self, mock_engine) -> None:
        self._seed_mentions(mock_engine.storage)
        with patch("src.api.app_config.USE_SOURCE_IDS_FILTER", True):
            with patch("src.api._get_engine", return_value=mock_engine):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    resp = await client.get(
                        "/api/geo/mentions",
                        params={
                            "source_ids": "",
                            "min_confidence": 0.0,
                        },
                    )

        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 0
        assert body["mentions"] == []

    @pytest.mark.anyio
    async def test_no_source_filters_preserves_all_sources_behavior(self, mock_engine) -> None:
        self._seed_mentions(mock_engine.storage)
        with patch("src.api.app_config.USE_SOURCE_IDS_FILTER", True):
            with patch("src.api._get_engine", return_value=mock_engine):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    resp = await client.get(
                        "/api/geo/mentions",
                        params={"min_confidence": 0.0},
                    )

        assert resp.status_code == 200
        mentions = resp.json()["mentions"]
        all_sources = {sid for group in mentions for sid in group.get("source_ids", [])}
        assert "doc_a" in all_sources
        assert "doc_b" in all_sources

    @pytest.mark.anyio
    async def test_geo_mentions_dedupes_duplicate_rows_by_source_chunk_place(self, mock_engine) -> None:
        self._seed_mentions(mock_engine.storage)
        mock_engine.storage.upsert_geo_mentions(
            [
                {
                    "id": "m-doc-a-dup",
                    "source_id": "doc_a",
                    "chunk_id": "c-doc-a",
                    "place_name": "Paris",
                    "matched_input": "Paris",
                    "matched_on": "paris",
                    "geonameid": 2988507,
                    "lat": 48.8566,
                    "lon": 2.3522,
                    "confidence": 0.95,
                    "method": "exact",
                }
            ]
        )

        with patch("src.api.app_config.USE_SOURCE_IDS_FILTER", True):
            with patch("src.api._get_engine", return_value=mock_engine):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    resp = await client.get(
                        "/api/geo/mentions",
                        params={"min_confidence": 0.0, "detailed": True},
                    )

        assert resp.status_code == 200
        mentions = resp.json()["mentions"]
        paris = next(item for item in mentions if item["place_name"] == "Paris")
        assert paris["mention_count"] == 1

    @pytest.mark.anyio
    async def test_geo_mentions_q_filter_returns_matching_places(self, mock_engine) -> None:
        self._seed_mentions(mock_engine.storage)

        with patch("src.api.app_config.USE_SOURCE_IDS_FILTER", True):
            with patch("src.api._get_engine", return_value=mock_engine):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    resp = await client.get(
                        "/api/geo/mentions",
                        params={"min_confidence": 0.0, "q": "pari"},
                    )

        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 1
        assert body["mentions"][0]["place_name"] == "Paris"


# ---------------------------------------------------------------------------
# POST /api/sources/ingest
# ---------------------------------------------------------------------------


class TestIngestEndpoint:
    @pytest.mark.anyio
    async def test_ingest_success(self, mock_engine, sample_file) -> None:
        with patch("src.api._get_engine", return_value=mock_engine):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/api/sources/ingest",
                    json={
                        "file_path": str(sample_file),
                        "source_id": "new_doc",
                    },
                )

        assert resp.status_code == 200
        data = resp.json()
        assert data["source_id"] == "new_doc"
        assert data["parents_count"] >= 1
        assert data["summarized"] is True

    @pytest.mark.anyio
    async def test_ingest_file_not_found(self, mock_engine) -> None:
        with patch("src.api._get_engine", return_value=mock_engine):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/api/sources/ingest",
                    json={
                        "file_path": "/nonexistent/file.pdf",
                        "source_id": "missing",
                    },
                )

        assert resp.status_code == 404
        data = resp.json()
        assert data["error"]["code"] == "SOURCE_NOT_FOUND"

    @pytest.mark.anyio
    async def test_ingest_no_summarize(self, mock_engine, sample_file) -> None:
        with patch("src.api._get_engine", return_value=mock_engine):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/api/sources/ingest",
                    json={
                        "file_path": str(sample_file),
                        "source_id": "no_summary",
                        "summarize": False,
                    },
                )

        assert resp.status_code == 200
        assert resp.json()["summarized"] is False

    @pytest.mark.anyio
    async def test_ingest_accepts_peopletag(self, mock_engine, sample_file) -> None:
        with patch("src.api._get_engine", return_value=mock_engine):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/api/sources/ingest",
                    json={
                        "file_path": str(sample_file),
                        "source_id": "people_ingest_doc",
                        "peopletag": True,
                    },
                )

        assert resp.status_code == 200
        data = resp.json()
        assert data["source_id"] == "people_ingest_doc"

    @pytest.mark.anyio
    async def test_ingest_persists_citation_reference_in_source_list(self, mock_engine, sample_file) -> None:
        with patch("src.api._get_engine", return_value=mock_engine):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                ingest_resp = await client.post(
                    "/api/sources/ingest",
                    json={
                        "file_path": str(sample_file),
                        "source_id": "citation_doc",
                        "summarize": False,
                        "citation_reference": "Smith (2024) Citation",
                    },
                )
                assert ingest_resp.status_code == 200
                list_resp = await client.get("/api/sources")

        assert list_resp.status_code == 200
        sources = list_resp.json()["sources"]
        row = next(source for source in sources if source["source_id"] == "citation_doc")
        assert row["citation_reference"] == "Smith (2024) Citation"
        assert row["source_path"] is not None
        assert row["snapshot_path"] is not None

    @pytest.mark.anyio
    async def test_ingest_reports_degraded_ner_diagnostics(self, mock_engine, sample_file) -> None:
        mock_engine.simulate_ner_unavailable = True

        with patch("src.api._get_engine", return_value=mock_engine):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/api/sources/ingest",
                    json={
                        "file_path": str(sample_file),
                        "source_id": "degraded_ner_doc",
                        "summarize": False,
                        "geotag": True,
                        "peopletag": True,
                    },
                )

        assert resp.status_code == 200
        data = resp.json()
        assert data["source_id"] == "degraded_ner_doc"
        assert data["geotag_ner"]["ner_available"] is False
        assert data["geotag_ner"]["method"] == "regex_fallback"
        assert data["peopletag_ner"]["ner_available"] is False
        assert data["peopletag_ner"]["method"] == "empty"


# ---------------------------------------------------------------------------
# POST /api/sources/upload
# ---------------------------------------------------------------------------


class TestUploadEndpoint:
    @pytest.mark.anyio
    async def test_upload_rejects_duplicate_source_id(self, mock_engine, sample_file) -> None:
        mock_engine.ingest(str(sample_file), source_id="dup_doc")

        with patch("src.api._get_engine", return_value=mock_engine):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                with sample_file.open("rb") as handle:
                    resp = await client.post(
                        "/api/sources/upload",
                        files={"file": ("sample.md", handle, "text/markdown")},
                        data={
                            "source_id": "dup_doc",
                            "summarize": "true",
                            "geotag": "false",
                            "page_offset": "1",
                        },
                    )

        assert resp.status_code == 409
        data = resp.json()
        assert data["error"]["code"] == "SOURCE_ALREADY_EXISTS"

    @pytest.mark.anyio
    async def test_upload_rejects_duplicate_when_source_id_is_auto_derived(self, mock_engine, sample_file) -> None:
        # source_id auto-derives from filename stem: sample.md -> sample
        mock_engine.ingest(str(sample_file), source_id="sample")

        with patch("src.api._get_engine", return_value=mock_engine):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                with sample_file.open("rb") as handle:
                    resp = await client.post(
                        "/api/sources/upload",
                        files={"file": ("sample.md", handle, "text/markdown")},
                        data={
                            "source_id": "",
                            "summarize": "true",
                            "geotag": "false",
                            "page_offset": "1",
                        },
                    )

        assert resp.status_code == 409
        data = resp.json()
        assert data["error"]["code"] == "SOURCE_ALREADY_EXISTS"

    @pytest.mark.anyio
    async def test_upload_accepts_peopletag_form_field(self, mock_engine, sample_file) -> None:
        with patch("src.api._get_engine", return_value=mock_engine):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                with sample_file.open("rb") as handle:
                    resp = await client.post(
                        "/api/sources/upload",
                        files={"file": ("sample.md", handle, "text/markdown")},
                        data={
                            "source_id": "people_upload_doc",
                            "summarize": "true",
                            "geotag": "false",
                            "peopletag": "true",
                            "page_offset": "1",
                        },
                    )

        assert resp.status_code == 200
        assert resp.json()["source_id"] == "people_upload_doc"

    @pytest.mark.anyio
    async def test_upload_persists_citation_reference(self, mock_engine, sample_file) -> None:
        with patch("src.api._get_engine", return_value=mock_engine):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                with sample_file.open("rb") as handle:
                    upload_resp = await client.post(
                        "/api/sources/upload",
                        files={"file": ("sample.md", handle, "text/markdown")},
                        data={
                            "source_id": "upload_citation_doc",
                            "summarize": "false",
                            "citation_reference": "Doe (2025) Upload Citation",
                            "page_offset": "1",
                        },
                    )
                assert upload_resp.status_code == 200
                list_resp = await client.get("/api/sources")

        assert list_resp.status_code == 200
        sources = list_resp.json()["sources"]
        row = next(source for source in sources if source["source_id"] == "upload_citation_doc")
        assert row["citation_reference"] == "Doe (2025) Upload Citation"

    @pytest.mark.anyio
    async def test_upload_reports_degraded_ner_diagnostics(self, mock_engine, sample_file) -> None:
        mock_engine.simulate_ner_unavailable = True

        with patch("src.api._get_engine", return_value=mock_engine):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                with sample_file.open("rb") as handle:
                    resp = await client.post(
                        "/api/sources/upload",
                        files={"file": ("sample.md", handle, "text/markdown")},
                        data={
                            "source_id": "upload_degraded_ner",
                            "summarize": "false",
                            "geotag": "true",
                            "peopletag": "true",
                            "page_offset": "1",
                        },
                    )

        assert resp.status_code == 200
        data = resp.json()
        assert data["source_id"] == "upload_degraded_ner"
        assert data["geotag_ner"]["ner_available"] is False
        assert data["geotag_ner"]["method"] == "regex_fallback"
        assert data["peopletag_ner"]["ner_available"] is False
        assert data["peopletag_ner"]["method"] == "empty"


# ---------------------------------------------------------------------------
# DELETE /api/sources/{source_id}
# ---------------------------------------------------------------------------


class TestDeleteEndpoint:
    @pytest.mark.anyio
    async def test_delete_existing(self, mock_engine, sample_file) -> None:
        mock_engine.ingest(str(sample_file), source_id="to_delete")

        with patch("src.api._get_engine", return_value=mock_engine):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.delete("/api/sources/to_delete")

        assert resp.status_code == 200
        data = resp.json()
        assert data["source_id"] == "to_delete"
        assert data["deleted"] is True

    @pytest.mark.anyio
    async def test_delete_nonexistent(self, mock_engine) -> None:
        with patch("src.api._get_engine", return_value=mock_engine):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.delete("/api/sources/nonexistent")

        assert resp.status_code == 200
        data = resp.json()
        assert data["deleted"] is False

    @pytest.mark.anyio
    async def test_list_empty_after_delete(self, mock_engine, sample_file) -> None:
        """After deleting the only source, list should be empty."""
        mock_engine.ingest(str(sample_file), source_id="only_doc")

        with patch("src.api._get_engine", return_value=mock_engine):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                await client.delete("/api/sources/only_doc")
                resp = await client.get("/api/sources")

        sources = resp.json()["sources"]
        assert not any(s["source_id"] == "only_doc" for s in sources)


# ---------------------------------------------------------------------------
# GET /api/sources/{source_id}/content
# ---------------------------------------------------------------------------


class TestContentEndpoint:
    @pytest.mark.anyio
    async def test_content_not_found(self, mock_engine) -> None:
        with patch("src.api._get_engine", return_value=mock_engine):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get("/api/sources/nonexistent/content")

        assert resp.status_code == 404
        data = resp.json()
        assert data["error"]["code"] == "SOURCE_NOT_FOUND"

    @pytest.mark.anyio
    async def test_content_from_original(self, mock_engine, sample_file) -> None:
        """Content resolved from original file when it still exists."""
        mock_engine.ingest(str(sample_file), source_id="content_test")
        # Update with source_path pointing to the still-existing file
        mock_engine.storage.upsert_source_summary(
            source_id="content_test",
            summary="Summary",
            source_path=str(sample_file),
        )

        with patch("src.api._get_engine", return_value=mock_engine):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get("/api/sources/content_test/content")

        assert resp.status_code == 200
        data = resp.json()
        assert data["content_source"] == "original"
        assert "Test Document" in data["content"]

    @pytest.mark.anyio
    async def test_content_snapshot_fallback(self, mock_engine, sample_file, tmp_path) -> None:
        """When original is gone, content falls back to snapshot."""
        mock_engine.ingest(str(sample_file), source_id="snap_test")

        # Create snapshot
        snapshot_path = save_snapshot(
            "snap_test",
            "Cached snapshot content",
            cache_dir=tmp_path / "cache",
        )

        # Update summary with snapshot path and non-existent source path
        mock_engine.storage.upsert_source_summary(
            source_id="snap_test",
            summary="Summary",
            source_path="/gone/file.txt",
            snapshot_path=snapshot_path,
        )

        with patch("src.api._get_engine", return_value=mock_engine):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get("/api/sources/snap_test/content")

        assert resp.status_code == 200
        data = resp.json()
        assert data["content_source"] == "snapshot"
        assert data["content"] == "Cached snapshot content"

    @pytest.mark.anyio
    async def test_content_parent_text_fallback(self, mock_engine, sample_file) -> None:
        """When both original and snapshot are gone, parent texts serve as fallback."""
        mock_engine.ingest(str(sample_file), source_id="parent_test")
        # Update with non-existent paths
        mock_engine.storage.upsert_source_summary(
            source_id="parent_test",
            summary="Summary",
            source_path="/gone/file.txt",
            snapshot_path="/gone/snapshot.txt",
        )

        with patch("src.api._get_engine", return_value=mock_engine):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get("/api/sources/parent_test/content")

        assert resp.status_code == 200
        data = resp.json()
        assert data["content_source"] == "summary"
        assert "Test Document" in data["content"]

    @pytest.mark.anyio
    async def test_content_404_after_delete(self, mock_engine, sample_file) -> None:
        """After deleting a source, /content returns 404."""
        mock_engine.ingest(str(sample_file), source_id="delete_content_test")

        with patch("src.api._get_engine", return_value=mock_engine):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                # Delete the source
                await client.delete("/api/sources/delete_content_test")
                # Try to get content
                resp = await client.get("/api/sources/delete_content_test/content")

        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Negative path: ingest → move original → snapshot fallback → delete → 404
# ---------------------------------------------------------------------------


class TestNegativePath:
    @pytest.mark.anyio
    async def test_full_negative_path(self, mock_engine, tmp_path) -> None:
        """
        Full negative-path test as specified in the plan:
        1. Ingest source
        2. Move original file
        3. Confirm snapshot fallback
        4. Delete source
        5. Confirm /content returns 404
        """
        # Step 1: Create and ingest a file
        original = tmp_path / "paper.md"
        original.write_text("Original paper content for negative path test.", encoding="utf-8")
        mock_engine.ingest(str(original), source_id="neg_test")

        # Create snapshot
        snapshot_path = save_snapshot(
            "neg_test",
            "Original paper content for negative path test.",
            cache_dir=tmp_path / "cache",
        )
        mock_engine.storage.upsert_source_summary(
            source_id="neg_test",
            summary="Summary of neg_test",
            source_path=str(original),
            snapshot_path=snapshot_path,
        )

        with patch("src.api._get_engine", return_value=mock_engine):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                # Step 2: Snapshot is preferred when available
                resp = await client.get("/api/sources/neg_test/content")
                assert resp.status_code == 200
                assert resp.json()["content_source"] == "snapshot"

                # Step 3: Move (delete) original file
                original.unlink()

                # Step 4: Confirm snapshot fallback
                resp = await client.get("/api/sources/neg_test/content")
                assert resp.status_code == 200
                data = resp.json()
                assert data["content_source"] == "snapshot"
                assert "negative path test" in data["content"]

                # Step 5: Delete source
                resp = await client.delete("/api/sources/neg_test")
                assert resp.status_code == 200
                assert resp.json()["deleted"] is True

                # Step 6: Confirm /content returns 404
                resp = await client.get("/api/sources/neg_test/content")
                assert resp.status_code == 404
                assert resp.json()["error"]["code"] == "SOURCE_NOT_FOUND"


# ---------------------------------------------------------------------------
# Chunk endpoints include page range fields
# ---------------------------------------------------------------------------


class TestChunkRangeEndpoints:
    @pytest.mark.anyio
    async def test_chunk_detail_includes_range_fields(self, mock_engine, tmp_storage, mock_embedder) -> None:
        from src.models import ChildChunk, Metadata, ParentChunk

        parent = ParentChunk(
            id="p-api-range",
            text="[Page 20] Parent text",
            metadata=Metadata(
                source_id="api-range-doc",
                page_number=20,
                start_page=20,
                end_page=21,
                page_label="20",
                display_page="20",
                header_path="Document",
                parent_id=None,
            ),
        )
        child = ChildChunk(
            id="c-api-range",
            text="[Page 20] Child text",
            metadata=Metadata(
                source_id="api-range-doc",
                page_number=20,
                start_page=20,
                end_page=20,
                page_label="20",
                display_page="20",
                header_path="Document",
                parent_id=parent.id,
            ),
        )

        tmp_storage.add_parents([parent])
        emb = mock_embedder.encode([child.text], normalize_embeddings=True)
        tmp_storage.add_children([child], embeddings=emb)

        with patch("src.api._get_engine", return_value=mock_engine):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get("/api/sources/api-range-doc/chunk/c-api-range")

        assert resp.status_code == 200
        data = resp.json()
        assert data["page_number"] == 20
        assert data["start_page"] == 20
        assert data["end_page"] == 20

    @pytest.mark.anyio
    async def test_chunk_batch_includes_range_fields(self, mock_engine, tmp_storage, mock_embedder) -> None:
        from src.models import ChildChunk, Metadata, ParentChunk

        parent = ParentChunk(
            id="p-api-batch",
            text="Parent batch text",
            metadata=Metadata(
                source_id="api-batch-doc",
                page_number=30,
                start_page=30,
                end_page=31,
                header_path="Document",
                parent_id=None,
            ),
        )
        child = ChildChunk(
            id="c-api-batch",
            text="Batch child text",
            metadata=Metadata(
                source_id="api-batch-doc",
                page_number=31,
                start_page=31,
                end_page=31,
                header_path="Document",
                parent_id=parent.id,
            ),
        )

        tmp_storage.add_parents([parent])
        emb = mock_embedder.encode([child.text], normalize_embeddings=True)
        tmp_storage.add_children([child], embeddings=emb)

        with patch("src.api._get_engine", return_value=mock_engine):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get("/api/sources/api-batch-doc/chunks?ids=c-api-batch")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["chunks"]) == 1
        row = data["chunks"][0]
        assert row["page_number"] == 31
        assert row["start_page"] == 31
        assert row["end_page"] == 31
