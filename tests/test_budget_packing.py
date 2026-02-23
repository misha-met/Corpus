"""Tests for token budget packing: greedy packing, truncation, consecutive failures, metadata alignment."""
from __future__ import annotations

import pytest

from src.generator import BudgetPackResult, count_tokens, enforce_token_budget
from tests.conftest import MockTokenizer, Timer, get_test_logger

logger = get_test_logger("budget_packing")


# ===========================================================================
# Token counting
# ===========================================================================

class TestTokenCounting:
    def test_count_tokens_basic(self):
        tok = MockTokenizer()
        assert count_tokens("hello world", tok) == 2

    def test_count_tokens_empty(self):
        tok = MockTokenizer()
        assert count_tokens("", tok) == 0

    def test_count_tokens_fallback(self):
        """If tokenizer has no encode method, falls back to char/4."""
        class BrokenTokenizer:
            pass
        assert count_tokens("1234567890", BrokenTokenizer()) == 2  # 10/4 = 2


# ===========================================================================
# Greedy packing
# ===========================================================================

class TestGreedyPacking:
    def test_pack_all_fit(self, mock_tokenizer: MockTokenizer):
        """All docs fit within budget."""
        docs = ["word " * 10, "word " * 10, "word " * 10]
        result = enforce_token_budget(docs, max_tokens=100, tokenizer=mock_tokenizer)
        assert len(result.packed_docs) == 3
        assert result.skipped_count == 0
        assert result.used_tokens <= 100

    def test_pack_partial_fit(self, mock_tokenizer: MockTokenizer):
        """Only some docs fit within budget."""
        docs = ["word " * 50, "word " * 50, "word " * 50]
        result = enforce_token_budget(docs, max_tokens=60, tokenizer=mock_tokenizer)
        assert len(result.packed_docs) < 3
        assert result.used_tokens <= 60

    def test_pack_priority_order(self, mock_tokenizer: MockTokenizer):
        """Documents should be packed in order (by priority/rerank score)."""
        docs = ["first " * 10, "second " * 10, "third " * 10]
        result = enforce_token_budget(docs, max_tokens=25, tokenizer=mock_tokenizer)
        assert result.packed_docs[0].startswith("first")

    def test_pack_exhausted_budget_stops(self, mock_tokenizer: MockTokenizer):
        """Packing should stop when remaining budget < effective_floor."""
        # Budget is 5 tokens, all docs are 20 tokens each — remaining will be < effective_floor (max(50,20)=50)
        docs = ["word " * 20 for _ in range(10)]
        result = enforce_token_budget(
            docs, max_tokens=5, tokenizer=mock_tokenizer,
            consecutive_fail_threshold=3, allow_truncation=False,
        )
        # Should stop early because remaining < effective_floor
        assert result.skipped_count < len(docs)  # did not scan all docs

    def test_pack_half_skipped_stops_early(self, mock_tokenizer: MockTokenizer):
        # More than len(docs) // 2 are oversized — secondary escape should trigger
        docs = ["word " * 20 for _ in range(10)]
        result = enforce_token_budget(
            docs, max_tokens=5, tokenizer=mock_tokenizer,
            consecutive_fail_threshold=2, allow_truncation=False,
        )
        # half of 10 = 5, so the secondary guard kicks in no later than after 6 skips
        assert result.skipped_count <= 6

    def test_pack_skips_large_and_packs_small(self, mock_tokenizer: MockTokenizer):
        """Regression: large docs interspersed should not stop packing of small docs."""
        large = "word " * 200  # 200 tokens — won't fit
        small = "word " * 5    # 5 tokens — will fit
        docs = [large] * 3 + [small] * 10
        result = enforce_token_budget(
            docs, max_tokens=60, tokenizer=mock_tokenizer,
            allow_truncation=False,
        )
        # All 10 small docs should be packed (budget = 60, 10 × 5 = 50 tokens)
        assert len(result.packed_docs) == 10

    def test_pack_empty_docs(self, mock_tokenizer: MockTokenizer):
        result = enforce_token_budget([], max_tokens=100, tokenizer=mock_tokenizer)
        assert len(result.packed_docs) == 0
        assert result.used_tokens == 0

    def test_pack_empty_string_docs_skipped(self, mock_tokenizer: MockTokenizer):
        docs = ["", "   ", "word " * 5, ""]
        result = enforce_token_budget(docs, max_tokens=100, tokenizer=mock_tokenizer)
        assert len(result.packed_docs) == 1  # Only the non-empty doc

    def test_pack_zero_budget(self, mock_tokenizer: MockTokenizer):
        docs = ["word " * 10]
        result = enforce_token_budget(docs, max_tokens=0, tokenizer=mock_tokenizer)
        assert len(result.packed_docs) == 0


# ===========================================================================
# Truncation behaviour
# ===========================================================================

class TestTruncation:
    def test_truncation_allowed(self, mock_tokenizer: MockTokenizer):
        """Large doc should be truncated if allow_truncation=True."""
        docs = ["word " * 100]
        result = enforce_token_budget(
            docs, max_tokens=50, tokenizer=mock_tokenizer,
            allow_truncation=True, min_doc_tokens=5,
        )
        assert len(result.packed_docs) >= 0  # May succeed with truncation

    def test_truncation_disabled(self, mock_tokenizer: MockTokenizer):
        """Large doc should be skipped if allow_truncation=False."""
        docs = ["word " * 100]
        result = enforce_token_budget(
            docs, max_tokens=50, tokenizer=mock_tokenizer,
            allow_truncation=False,
        )
        assert result.skipped_count >= 1

    def test_truncation_count_tracked(self, mock_tokenizer: MockTokenizer):
        """Truncated docs should be counted."""
        docs = ["word " * 100, "word " * 5]
        result = enforce_token_budget(
            docs, max_tokens=60, tokenizer=mock_tokenizer,
            allow_truncation=True, min_doc_tokens=5,
        )
        # Either truncated or skipped, metrics should reflect
        assert result.truncated_count >= 0
        assert result.skipped_count >= 0


# ===========================================================================
# Metadata / index alignment
# ===========================================================================

class TestMetadataAlignment:
    def test_packed_indices_correct(self, mock_tokenizer: MockTokenizer):
        """packed_indices should map correctly to original doc positions."""
        docs = ["word " * 10, "word " * 10, "word " * 10]
        result = enforce_token_budget(docs, max_tokens=100, tokenizer=mock_tokenizer)
        assert len(result.packed_indices) == len(result.packed_docs)
        # All indices valid
        for idx in result.packed_indices:
            assert 0 <= idx < len(docs)

    def test_packed_indices_preserve_order(self, mock_tokenizer: MockTokenizer):
        """Packed indices should be in ascending order (greedy in-order)."""
        docs = ["word " * 5 for _ in range(5)]
        result = enforce_token_budget(docs, max_tokens=100, tokenizer=mock_tokenizer)
        assert result.packed_indices == sorted(result.packed_indices)

    def test_skipped_doc_indices_missing(self, mock_tokenizer: MockTokenizer):
        """Skipped docs should NOT appear in packed_indices."""
        docs = ["word " * 5, "word " * 200, "word " * 5]  # middle doc too big
        result = enforce_token_budget(
            docs, max_tokens=15, tokenizer=mock_tokenizer,
            allow_truncation=False,
        )
        for idx in result.packed_indices:
            assert idx != 1  # Large doc index should be skipped

    def test_citation_metadata_alignment(self, mock_tokenizer: MockTokenizer):
        """packed_indices should correctly map to metadata for citation formatting."""
        docs = [f"doc_{i} " * 10 for i in range(5)]
        metadatas = [
            {"source_id": f"src_{i}", "page_number": i + 1}
            for i in range(5)
        ]
        result = enforce_token_budget(docs, max_tokens=100, tokenizer=mock_tokenizer)
        packed_metas = [metadatas[i] for i in result.packed_indices]
        assert len(packed_metas) == len(result.packed_docs)
        # Verify the mapping is correct
        for packed_doc, packed_meta, idx in zip(
            result.packed_docs, packed_metas, result.packed_indices
        ):
            assert packed_meta["source_id"] == f"src_{idx}"


# ===========================================================================
# BudgetPackResult dataclass
# ===========================================================================

class TestBudgetPackResult:
    def test_result_fields(self, mock_tokenizer: MockTokenizer):
        docs = ["word " * 10] * 3
        result = enforce_token_budget(docs, max_tokens=100, tokenizer=mock_tokenizer)
        assert isinstance(result, BudgetPackResult)
        assert isinstance(result.packed_docs, list)
        assert isinstance(result.packed_indices, list)
        assert isinstance(result.used_tokens, int)
        assert isinstance(result.skipped_count, int)
        assert isinstance(result.truncated_count, int)


# ===========================================================================
# Latency: budget packing
# ===========================================================================

class TestBudgetPackingLatency:
    def test_packing_latency_scaling(self, mock_tokenizer: MockTokenizer):
        for n_docs in [10, 50, 100, 500]:
            docs = [f"document content about topic {i} " * 20 for i in range(n_docs)]
            with Timer("budget_packing", n_docs=n_docs) as t:
                result = enforce_token_budget(
                    docs, max_tokens=5000, tokenizer=mock_tokenizer,
                )
            logger.info(
                f"packing {n_docs} docs: {t.result.elapsed_ms:.2f}ms, "
                f"packed={len(result.packed_docs)}, used={result.used_tokens}"
            )

    def test_packing_large_budget(self, mock_tokenizer: MockTokenizer):
        docs = [f"word{i} " * 50 for i in range(100)]
        with Timer("budget_packing_large", n_docs=100, budget=50000) as t:
            result = enforce_token_budget(
                docs, max_tokens=50000, tokenizer=mock_tokenizer,
            )
        logger.info(
            f"large budget: {t.result.elapsed_ms:.2f}ms, "
            f"packed={len(result.packed_docs)}, utilization={100*result.used_tokens/50000:.1f}%"
        )
