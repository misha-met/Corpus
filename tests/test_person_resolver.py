from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from src.person_resolver import PersonResolver


def _resolver() -> PersonResolver:
    return PersonResolver(
        fuzzy_threshold_lastname=80,
        fuzzy_threshold_fullname=80,
    )


def test_normalize_name_strips_titles() -> None:
    resolver = _resolver()
    assert resolver.normalize_name("Dr. Noam Chomsky") == "noam chomsky"
    assert resolver.normalize_name("  Prof  Jane  Doe ") == "jane doe"


def test_matching_order_exact_then_fuzzy() -> None:
    resolver = _resolver()

    first = resolver.resolve(
        raw_name="Noam Chomsky",
        source_id="doc_a",
        ner_score=0.95,
        context_words=["written", "by"],
        context_snippet="written by Noam Chomsky",
    )
    assert first is not None
    assert first["method"] == "new"

    second = resolver.resolve(
        raw_name="Chomsky",
        source_id="doc_b",
        ner_score=0.92,
        context_words=["according", "to"],
        context_snippet="according to Chomsky",
    )
    assert second is not None
    assert second["canonical_name"] == "Noam Chomsky"
    assert second["method"] == "fuzzy_last"

    third = resolver.resolve(
        raw_name="Prof. Noam Chomsky",
        source_id="doc_c",
        ner_score=0.90,
        context_words=["author"],
        context_snippet="author Noam Chomsky",
    )
    assert third is not None
    assert third["canonical_name"] == "Noam Chomsky"
    assert third["method"] == "exact"


def test_canonical_promotion_prefers_more_informative_variant() -> None:
    resolver = _resolver()

    seed = resolver.resolve(
        raw_name="Chomsky",
        source_id="doc_a",
        ner_score=0.9,
        context_words=[],
        context_snippet="",
    )
    assert seed is not None
    assert seed["canonical_name"] == "Chomsky"

    promoted = resolver.resolve(
        raw_name="Noam Chomsky",
        source_id="doc_a",
        ner_score=0.91,
        context_words=[],
        context_snippet="",
    )
    assert promoted is not None
    assert promoted["canonical_name"] == "Noam Chomsky"

    snapshot = resolver.snapshot()
    assert "Noam Chomsky" in snapshot
    assert "Chomsky" not in snapshot


def test_role_inference_taxonomy() -> None:
    assert PersonResolver.infer_role_hint(context_words=["written", "by"], context_snippet="written by John Doe") == "author"
    assert PersonResolver.infer_role_hint(context_words=["according", "to"], context_snippet="according to John Doe") == "cited"
    assert PersonResolver.infer_role_hint(context_words=["focuses", "on"], context_snippet="focuses on John Doe") == "subject"
    assert PersonResolver.infer_role_hint(context_words=[], context_snippet="") == "mentioned"


def test_remove_mention_decrements_and_removes_entry() -> None:
    resolver = _resolver()

    resolver.resolve(
        raw_name="Noam Chomsky",
        source_id="doc_a",
        ner_score=0.95,
        context_words=[],
        context_snippet="",
    )
    resolver.resolve(
        raw_name="Noam Chomsky",
        source_id="doc_b",
        ner_score=0.95,
        context_words=[],
        context_snippet="",
    )

    resolver.remove_mention(
        canonical_name="Noam Chomsky",
        raw_name="Noam Chomsky",
        source_id="doc_a",
    )
    snap = resolver.snapshot()
    assert snap["Noam Chomsky"]["mention_count"] == 1

    resolver.remove_mention(
        canonical_name="Noam Chomsky",
        raw_name="Noam Chomsky",
        source_id="doc_b",
    )
    assert resolver.snapshot() == {}


def test_warm_from_rows_rebuilds_registry() -> None:
    resolver = _resolver()
    resolver.warm_from_rows(
        [
            {"source_id": "doc_a", "raw_name": "Noam Chomsky", "canonical_name": "Noam Chomsky"},
            {"source_id": "doc_b", "raw_name": "Chomsky", "canonical_name": "Noam Chomsky"},
        ]
    )
    snap = resolver.snapshot()
    assert snap["Noam Chomsky"]["mention_count"] == 2
    assert sorted(snap["Noam Chomsky"]["source_ids"]) == ["doc_a", "doc_b"]


def test_thread_safe_resolve_updates() -> None:
    resolver = _resolver()

    def _work(_idx: int) -> None:
        for _ in range(25):
            resolver.resolve(
                raw_name="Noam Chomsky",
                source_id="doc_thread",
                ner_score=0.9,
                context_words=[],
                context_snippet="",
            )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(_work, range(8)))

    snap = resolver.snapshot()
    assert snap["Noam Chomsky"]["mention_count"] == 200
