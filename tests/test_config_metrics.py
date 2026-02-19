"""Tests for configuration system and metrics computation."""
from __future__ import annotations

import os
from unittest import mock

import pytest

from src.config import (
    VALID_MODES,
    MODE_RAM_REQUIREMENTS,
    ModelConfig,
    _auto_select_mode,
    _get_mode_config,
    select_mode_config,
)
from src.metrics import (
    BudgetMetrics,
    DeduplicationMetrics,
    RerankerMetrics,
    RetrievalMetrics,
    ThresholdMetrics,
    TimingMetrics,
    compute_reranker_stats,
    format_metrics_summary,
)
from tests.conftest import get_test_logger

logger = get_test_logger("config_metrics")


# ===========================================================================
# Mode configuration
# ===========================================================================

class TestModeConfig:
    def test_all_modes_valid(self):
        """All advertised modes should produce valid configs."""
        for mode in VALID_MODES:
            config = _get_mode_config(mode, ram_gb=64.0)
            assert config.mode == mode
            assert config.llm_model
            assert config.embedding_model
            assert config.reranker_model

    def test_regular_mode_32gb(self):
        config = _get_mode_config("regular", ram_gb=32.0)
        assert config.context_window == 16_000  # Reduced for 32GB
        assert config.retrieval_budget == 8_000

    def test_regular_mode_48gb(self):
        """48-63GB systems get standard regular config."""
        config = _get_mode_config("regular", ram_gb=48.0)
        assert config.context_window == 64_000
        assert config.retrieval_budget == 32_000
        assert config.top_k_dense == 100
        assert config.top_k_rerank == 20
        assert config.top_k_final == 5

    def test_regular_mode_64gb(self):
        """64GB+ 'Regular Plus': deeper retrieval exploiting M4 Max bandwidth."""
        config = _get_mode_config("regular", ram_gb=64.0)
        assert config.context_window == 64_000
        assert config.retrieval_budget == 48_000
        assert config.top_k_dense == 200
        assert config.top_k_sparse == 200
        assert config.top_k_fused == 100
        assert config.top_k_rerank == 40
        assert config.top_k_final == 8
        assert config.reranker_threshold == 0.04
        assert config.reranker_min_docs == 4

    def test_power_deep_research_mode(self):
        """M4 Max 64GB calibration: aggressive but memory-safe."""
        config = _get_mode_config("power-deep-research", ram_gb=64.0)
        assert "80B" in config.llm_model or "Next" in config.llm_model
        # Context sized for ~10GB KV cache, leaving 2GB OS buffer
        assert config.context_window == 48_000
        assert config.retrieval_budget == 40_000
        # Wide initial retrieval, selective reranking
        assert config.top_k_dense == 400
        assert config.top_k_sparse == 400
        assert config.top_k_fused == 200
        assert config.top_k_rerank == 60
        # final docs matched to Regular mode's signal-to-noise sweet spot
        assert config.top_k_final == 8
        assert config.reranker_threshold == 0.04
        assert config.reranker_min_docs == 4

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            _get_mode_config("nonexistent", ram_gb=64.0)

    def test_mode_ram_requirements_present(self):
        for mode in VALID_MODES:
            assert mode in MODE_RAM_REQUIREMENTS


# ===========================================================================
# Auto mode selection
# ===========================================================================

class TestAutoModeSelection:
    def test_auto_low_ram(self):
        assert _auto_select_mode(32.0) == "regular"

    def test_auto_high_ram(self):
        # Auto-select always returns 'regular'; RAM-aware scaling happens in _get_mode_config
        assert _auto_select_mode(64.0) == "regular"

    def test_auto_boundary(self):
        # Both RAM tiers auto-select to 'regular' with RAM-aware parameter scaling
        assert _auto_select_mode(63.9) == "regular"
        assert _auto_select_mode(64.0) == "regular"


# ===========================================================================
# Mode selection precedence
# ===========================================================================

class TestModeSelectionPrecedence:
    def test_cli_overrides_env(self):
        with mock.patch.dict(os.environ, {"RAG_MODE": "power-deep-research"}):
            config = select_mode_config(manual_mode="regular")
            assert config.mode == "regular"

    def test_env_used_when_no_cli(self):
        with mock.patch.dict(os.environ, {"RAG_MODE": "power-deep-research"}):
            config = select_mode_config(manual_mode=None)
            assert config.mode == "power-deep-research"

    def test_auto_when_no_override(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            # Remove RAG_MODE if present
            env = os.environ.copy()
            env.pop("RAG_MODE", None)
            with mock.patch.dict(os.environ, env, clear=True):
                config = select_mode_config(manual_mode=None)
                assert config.mode in VALID_MODES

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            select_mode_config(manual_mode="super-turbo")

    def test_legacy_mapping(self):
        # Legacy modes should be mapped to regular
        with mock.patch.dict(os.environ, {}, clear=True):
            env = os.environ.copy()
            env.pop("RAG_MODE", None)
            with mock.patch.dict(os.environ, env, clear=True):
                config = select_mode_config(manual_mode="high")
                assert config.mode == "regular"

    def test_power_fast_legacy_mapping(self):
        # power-fast is now deprecated and maps to regular
        with mock.patch.dict(os.environ, {}, clear=True):
            env = os.environ.copy()
            env.pop("RAG_MODE", None)
            with mock.patch.dict(os.environ, env, clear=True):
                config = select_mode_config(manual_mode="power-fast")
                assert config.mode == "regular"


# ===========================================================================
# ModelConfig dataclass
# ===========================================================================

class TestModelConfig:
    def test_frozen_config(self):
        config = _get_mode_config("regular", 64.0)
        with pytest.raises(Exception):
            config.mode = "modified"  # type: ignore[misc]

    def test_retrieval_params_differ_by_mode(self):
        regular = _get_mode_config("regular", 64.0)
        deep = _get_mode_config("power-deep-research", 64.0)
        # Deep research mode has wider initial retrieval net
        assert deep.top_k_dense > regular.top_k_dense
        assert deep.top_k_rerank > regular.top_k_rerank

    def test_thresholds_differ_by_mode(self):
        regular = _get_mode_config("regular", 64.0)
        deep = _get_mode_config("power-deep-research", 64.0)
        # Deep research uses more permissive threshold for wider capture
        assert regular.reranker_threshold >= deep.reranker_threshold


# ===========================================================================
# Reranker stats computation
# ===========================================================================

class TestRerankerStats:
    def test_compute_basic(self):
        scores = [0.1, 0.5, 0.9]
        stats = compute_reranker_stats(scores)
        assert abs(stats.score_min - 0.1) < 1e-6
        assert abs(stats.score_max - 0.9) < 1e-6
        assert abs(stats.score_mean - 0.5) < 1e-6
        assert stats.items_reranked == 3

    def test_compute_single(self):
        stats = compute_reranker_stats([0.5])
        assert stats.score_std == 0.0
        assert stats.items_reranked == 1

    def test_compute_empty(self):
        stats = compute_reranker_stats([])
        assert stats.items_reranked == 0

    def test_compute_std(self):
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = compute_reranker_stats(scores)
        assert stats.score_std > 0


# ===========================================================================
# Metrics summary formatting
# ===========================================================================

class TestMetricsFormatting:
    def test_format_summary_basic(self):
        metrics = RetrievalMetrics(
            budget=BudgetMetrics(
                budget_tokens=10000,
                used_tokens=7500,
                utilization_pct=75.0,
                docs_packed=5,
            ),
            timing=TimingMetrics(total_ms=250.5),
        )
        summary = format_metrics_summary(metrics)
        assert "75%" in summary
        assert "250ms" in summary or "251ms" in summary
        assert "Docs: 5" in summary

    def test_format_summary_with_dedup(self):
        metrics = RetrievalMetrics(
            budget=BudgetMetrics(budget_tokens=10000, used_tokens=5000, utilization_pct=50.0, docs_packed=3),
            timing=TimingMetrics(total_ms=100.0),
            deduplication=DeduplicationMetrics(
                children_before_dedup=10,
                children_after_dedup=5,
                reduction_pct=50.0,
            ),
        )
        summary = format_metrics_summary(metrics)
        assert "Dedup" in summary


# ===========================================================================
# Timing metrics
# ===========================================================================

class TestTimingMetrics:
    def test_timing_defaults_zero(self):
        t = TimingMetrics()
        assert t.hybrid_search_ms == 0.0
        assert t.total_ms == 0.0

    def test_timing_all_stages(self):
        t = TimingMetrics(
            hybrid_search_ms=10.0,
            sparse_search_ms=5.0,
            rrf_fusion_ms=1.0,
            rerank_ms=100.0,
            dedup_ms=0.5,
            budget_packing_ms=2.0,
            total_ms=118.5,
        )
        assert t.rerank_ms == 100.0
        # Total should be sum of stages (approximately)
        stage_sum = t.hybrid_search_ms + t.sparse_search_ms + t.rrf_fusion_ms + t.rerank_ms + t.dedup_ms + t.budget_packing_ms
        assert abs(t.total_ms - stage_sum) < 1.0


# ===========================================================================
# Threshold metrics
# ===========================================================================

class TestThresholdMetrics:
    def test_threshold_defaults(self):
        t = ThresholdMetrics()
        assert t.safety_net_triggered is False

    def test_safety_net_recorded(self):
        t = ThresholdMetrics(
            threshold_value=0.05,
            items_before_threshold=10,
            items_after_threshold=3,
            safety_net_triggered=True,
            min_docs=3,
        )
        assert t.safety_net_triggered is True
