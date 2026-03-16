"""Tests for people dictionary API endpoints."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from src.api import app
from src.storage import StorageConfig, StorageEngine


class MockEngineWithStorage:
    def __init__(self, storage: StorageEngine):
        self._storage = storage

    @property
    def storage(self) -> StorageEngine:
        return self._storage

    def close(self) -> None:
        self._storage.close()


def _seed_person_mentions(storage: StorageEngine) -> None:
    storage.upsert_person_mentions(
        [
            {
                "id": "pm-1",
                "source_id": "doc_a",
                "chunk_id": "chunk-a1",
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
                "chunk_id": "chunk-b1",
                "raw_name": "Chomsky",
                "canonical_name": "Noam Chomsky",
                "confidence": 0.90,
                "method": "fuzzy_last",
                "role_hint": "cited",
                "context_snippet": "according to Chomsky",
            },
            {
                "id": "pm-3",
                "source_id": "doc_a",
                "chunk_id": "chunk-a2",
                "raw_name": "Michel Foucault",
                "canonical_name": "Michel Foucault",
                "confidence": 0.92,
                "method": "new",
                "role_hint": "subject",
                "context_snippet": "focuses on Michel Foucault",
            },
        ]
    )


@pytest.fixture
def tmp_storage(tmp_path: Path) -> StorageEngine:
    engine = StorageEngine(StorageConfig(lance_dir=tmp_path / "lance", lance_table="test_chunks"))
    yield engine
    engine.close()


@pytest.fixture
def mock_engine(tmp_storage: StorageEngine) -> MockEngineWithStorage:
    return MockEngineWithStorage(tmp_storage)


class TestPeopleListEndpoint:
    @pytest.mark.anyio
    async def test_people_list_returns_grouped_contract(self, mock_engine: MockEngineWithStorage) -> None:
        _seed_person_mentions(mock_engine.storage)
        with patch("src.api._get_engine", return_value=mock_engine):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                resp = await client.get("/api/people", params={"min_confidence": 0.0})

        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        noam = next(item for item in data["people"] if item["canonical_name"] == "Noam Chomsky")
        assert noam["mention_count"] == 2
        assert sorted(noam["source_ids"]) == ["doc_a", "doc_b"]
        assert "Chomsky" in noam["variants"]

    @pytest.mark.anyio
    async def test_source_union_and_explicit_empty_contract(self, mock_engine: MockEngineWithStorage) -> None:
        _seed_person_mentions(mock_engine.storage)

        with patch("src.api._get_engine", return_value=mock_engine):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                union_resp = await client.get(
                    "/api/people",
                    params={
                        "source_id": "doc_a",
                        "source_ids": ["doc_b"],
                        "min_confidence": 0.0,
                    },
                )
                empty_resp = await client.get(
                    "/api/people",
                    params={
                        "source_ids": "",
                        "min_confidence": 0.0,
                    },
                )

        assert union_resp.status_code == 200
        noam = next(item for item in union_resp.json()["people"] if item["canonical_name"] == "Noam Chomsky")
        assert noam["mention_count"] == 2

        assert empty_resp.status_code == 200
        assert empty_resp.json()["count"] == 0
        assert empty_resp.json()["people"] == []


class TestPeopleMentionsEndpoint:
    @pytest.mark.anyio
    async def test_mentions_by_canonical_with_source_filter(self, mock_engine: MockEngineWithStorage) -> None:
        _seed_person_mentions(mock_engine.storage)

        with patch("src.api._get_engine", return_value=mock_engine):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                resp = await client.get(
                    "/api/people/mentions",
                    params={
                        "canonical_name": "Noam Chomsky",
                        "source_ids": ["doc_a"],
                        "min_confidence": 0.0,
                    },
                )

        assert resp.status_code == 200
        body = resp.json()
        assert body["canonical_name"] == "Noam Chomsky"
        assert body["count"] == 1
        assert body["mentions"][0]["source_id"] == "doc_a"


class TestPeopleDeleteEndpoint:
    @pytest.mark.anyio
    async def test_delete_mention_returns_204_and_removes_row(self, mock_engine: MockEngineWithStorage) -> None:
        _seed_person_mentions(mock_engine.storage)

        with patch("src.api._get_engine", return_value=mock_engine):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                delete_resp = await client.delete("/api/people/mentions/pm-2")
                list_resp = await client.get(
                    "/api/people/mentions",
                    params={
                        "canonical_name": "Noam Chomsky",
                        "min_confidence": 0.0,
                    },
                )

        assert delete_resp.status_code == 204
        assert mock_engine.storage.get_person_mention("pm-2") is None
        assert list_resp.status_code == 200
        assert list_resp.json()["count"] == 1


class TestPeopleMergeEndpoint:
    @pytest.mark.anyio
    async def test_merge_canonical_names_updates_people_groups(self, mock_engine: MockEngineWithStorage) -> None:
        _seed_person_mentions(mock_engine.storage)

        with patch("src.api._get_engine", return_value=mock_engine):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                merge_resp = await client.post(
                    "/api/people/merge",
                    json={
                        "source_canonical_name": "Michel Foucault",
                        "target_canonical_name": "Noam Chomsky",
                    },
                )
                list_resp = await client.get("/api/people", params={"min_confidence": 0.0})

        assert merge_resp.status_code == 200
        merge_body = merge_resp.json()
        assert merge_body["source_canonical_name"] == "Michel Foucault"
        assert merge_body["target_canonical_name"] == "Noam Chomsky"
        assert merge_body["merged_count"] == 1

        assert list_resp.status_code == 200
        list_body = list_resp.json()
        assert list_body["count"] == 1
        assert list_body["people"][0]["canonical_name"] == "Noam Chomsky"
        assert list_body["people"][0]["mention_count"] == 3
