from __future__ import annotations

import logging
import os
import platform
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

CITATIONS_ENABLED_DEFAULT: bool = False
_detected_ram_gb: Optional[float] = None


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for RAG pipeline models and retrieval parameters."""
    mode: str
    llm_model: str
    embedding_model: str
    reranker_model: str
    embedding_device: str = "cpu"
    quantization: str = "4-bit"
    context_window: int = 128_000
    retrieval_budget: int = 100_000
    top_k_dense: int = 100
    top_k_sparse: int = 100
    top_k_fused: int = 50
    top_k_rerank: int = 20
    top_k_final: int = 5
    reranker_threshold: float = 0.05
    reranker_min_docs: int = 3
    reranker_enabled: bool = True
    context_expansion_enabled: bool = True
    system_ram_gb: float = 0.0
    max_children_per_parent: int = 2


# ── Intent-aware parameter overrides ─────────────────────────────────────────


@dataclass(frozen=True)
class IntentRetrievalOverrides:
    top_k_dense_scale: float = 1.0
    top_k_fused_scale: float = 1.0
    top_k_rerank_scale: float = 1.0
    top_k_final_scale: float = 1.0
    reranker_threshold_scale: float = 1.0
    reranker_min_docs: Optional[int] = None     # absolute override, not scale


@dataclass(frozen=True)
class IntentGenerationParams:
    temperature: float
    top_p: float


@dataclass(frozen=True)
class ResolvedRetrievalParams:
    top_k_dense: int
    top_k_fused: int
    top_k_rerank: int
    top_k_final: int
    reranker_threshold: float
    reranker_min_docs: int
    max_children_per_parent: int = 2


INTENT_RETRIEVAL_OVERRIDES: dict[str, IntentRetrievalOverrides] = {
    # FACTUAL: stricter threshold (1.3×) + min_docs=1 so a single high-confidence
    # fact is not padded with low-quality backfill chunks.
    "FACTUAL":    IntentRetrievalOverrides(top_k_dense_scale=0.7, top_k_fused_scale=0.5, top_k_final_scale=1.0, reranker_threshold_scale=1.3, reranker_min_docs=1),
    "SUMMARIZE":  IntentRetrievalOverrides(top_k_dense_scale=1.0, top_k_final_scale=1.0),
    "OVERVIEW":   IntentRetrievalOverrides(top_k_dense_scale=1.2, top_k_final_scale=1.0),
    "EXPLAIN":    IntentRetrievalOverrides(top_k_dense_scale=1.0, top_k_final_scale=1.0),
    "ANALYZE":    IntentRetrievalOverrides(top_k_dense_scale=1.3, top_k_final_scale=1.5, reranker_threshold_scale=0.8),
    "COMPARE":    IntentRetrievalOverrides(top_k_dense_scale=1.3, top_k_final_scale=1.5, reranker_threshold_scale=0.8),
    "CRITIQUE":   IntentRetrievalOverrides(top_k_dense_scale=1.2, top_k_final_scale=1.25),
    "COLLECTION": IntentRetrievalOverrides(top_k_dense_scale=0.8, top_k_fused_scale=0.5, top_k_final_scale=1.0, reranker_threshold_scale=1.2),
}

INTENT_GENERATION_PARAMS: dict[str, IntentGenerationParams] = {
    "FACTUAL":    IntentGenerationParams(temperature=0.1, top_p=0.2),
    "SUMMARIZE":  IntentGenerationParams(temperature=0.3, top_p=0.6),
    "OVERVIEW":   IntentGenerationParams(temperature=0.3, top_p=0.6),
    "EXPLAIN":    IntentGenerationParams(temperature=0.4, top_p=0.7),
    "ANALYZE":    IntentGenerationParams(temperature=0.4, top_p=0.7),
    "COMPARE":    IntentGenerationParams(temperature=0.35, top_p=0.65),
    "CRITIQUE":   IntentGenerationParams(temperature=0.45, top_p=0.75),
    "COLLECTION": IntentGenerationParams(temperature=0.2, top_p=0.4),
}


def resolve_retrieval_params(mode_config: ModelConfig, intent: str) -> ResolvedRetrievalParams:
    """Apply intent-specific scale factors to base mode config values.

    Scale factors multiply the mode's base value. Absolute overrides (top_k_final,
    reranker_min_docs) replace the base value when not None. All scaled values are
    clamped to minimum 1. Falls back to base mode values if intent is not recognized.
    """
    overrides = INTENT_RETRIEVAL_OVERRIDES.get(intent.upper(), IntentRetrievalOverrides())
    return ResolvedRetrievalParams(
        top_k_dense=max(1, round(mode_config.top_k_dense * overrides.top_k_dense_scale)),
        top_k_fused=max(1, round(mode_config.top_k_fused * overrides.top_k_fused_scale)),
        top_k_rerank=max(1, round(mode_config.top_k_rerank * overrides.top_k_rerank_scale)),
        top_k_final=max(1, round(mode_config.top_k_final * overrides.top_k_final_scale)),
        reranker_threshold=mode_config.reranker_threshold * overrides.reranker_threshold_scale,
        reranker_min_docs=(
            overrides.reranker_min_docs
            if overrides.reranker_min_docs is not None
            else mode_config.reranker_min_docs
        ),
        max_children_per_parent=mode_config.max_children_per_parent,
    )


def resolve_generation_params(intent: str) -> IntentGenerationParams:
    """Return generation params for intent. Falls back to OVERVIEW if unrecognized."""
    return INTENT_GENERATION_PARAMS.get(intent.upper(), INTENT_GENERATION_PARAMS["OVERVIEW"])


def _get_mode_config(mode: str, ram_gb: float) -> ModelConfig:
    """Get mode configuration with RAM-aware token budget adjustments."""

    # Shared model stack per mode (fields identical across RAM tiers)
    _REGULAR_MODELS = dict(
        llm_model="mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit",
        embedding_model="mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
        reranker_model="jinaai/jina-reranker-v3-mlx",
        embedding_device="cpu",
        quantization="4-bit",
    )

    if mode == "regular":
        if ram_gb < 48:
            # 32GB systems: reduced context to avoid swap thrashing
            return ModelConfig(
                mode="regular",
                **_REGULAR_MODELS,
                context_window=16_000,
                retrieval_budget=8_000,
                system_ram_gb=ram_gb,
            )
        elif ram_gb >= 64:
            # M4 Max 64GB: deep retrieval exploiting 500GB/s+ bandwidth
            return ModelConfig(
                mode="regular",
                **_REGULAR_MODELS,
                context_window=64_000,
                retrieval_budget=48_000,
                top_k_dense=200,
                top_k_sparse=200,
                top_k_fused=100,
                top_k_rerank=40,
                top_k_final=8,
                reranker_threshold=0.04,
                reranker_min_docs=4,
                system_ram_gb=ram_gb,
            )
        else:
            # 48-63GB systems: standard context, moderate retrieval
            return ModelConfig(
                mode="regular",
                **_REGULAR_MODELS,
                context_window=64_000,
                retrieval_budget=32_000,
                system_ram_gb=ram_gb,
            )

    elif mode == "power-deep-research":
        if ram_gb < 64:
            logger.warning(f"power-deep-research mode requires 64GB+ RAM. Detected {ram_gb:.0f}GB.")
        return ModelConfig(
            mode="power-deep-research",
            llm_model="mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit",
            embedding_model="mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
            reranker_model="jinaai/jina-reranker-v3-mlx",
            embedding_device="cpu",
            quantization="4-bit",
            context_window=48_000,
            retrieval_budget=40_000,
            top_k_dense=400,
            top_k_sparse=400,
            top_k_fused=200,
            top_k_rerank=60,
            top_k_final=8,
            reranker_threshold=0.04,
            reranker_min_docs=4,
            reranker_enabled=True,
            context_expansion_enabled=True,
            system_ram_gb=ram_gb,
        )

    raise ValueError(f"Unknown mode: {mode}")


VALID_MODES = {"regular", "power-deep-research"}
MODE_RAM_REQUIREMENTS: dict[str, float] = {
    "regular": 32.0,
    "power-deep-research": 64.0,
}


# =============================================================================
# System Detection
# =============================================================================

def _detect_ram_gb() -> float:
    """Detect system RAM in gigabytes.
    
    Uses psutil if available for cross-platform accuracy,
    falls back to platform-specific methods.
    Caches result for subsequent calls.
    
    Returns:
        System RAM in GB, or 0.0 if detection fails.
    """
    global _detected_ram_gb
    if _detected_ram_gb is not None:
        return _detected_ram_gb
    
    ram_gb = 0.0
    
    # Try psutil first (most reliable)
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
        _detected_ram_gb = ram_gb
        return ram_gb
    except ImportError:
        pass
    except Exception as exc:
        logger.debug(f"psutil RAM detection failed: {exc}")

    # Fallback: macOS-specific
    system = platform.system().lower()
    if system == "darwin":
        try:
            import subprocess
            output = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            ram_gb = int(output.strip()) / (1024**3)
            _detected_ram_gb = ram_gb
            return ram_gb
        except Exception as exc:
            logger.debug(f"macOS RAM detection failed: {exc}")

    # Fallback: Linux sysconf
    if hasattr(os, "sysconf"):
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            ram_gb = (pages * page_size) / (1024**3)
            _detected_ram_gb = ram_gb
            return ram_gb
        except (ValueError, OSError, AttributeError) as exc:
            logger.debug(f"sysconf RAM detection failed: {exc}")

    _detected_ram_gb = ram_gb
    return ram_gb


def get_system_ram_gb() -> float:
    """Public accessor for detected system RAM (cached)."""
    return _detect_ram_gb()


def _auto_select_mode(ram_gb: float) -> str:
    # Both tiers (32GB and 64GB+) use 'regular' mode; RAM-aware config branching
    # happens inside _get_mode_config() to provide different parameters
    return "regular"


def select_mode_config(*, manual_mode: Optional[str] = None) -> ModelConfig:
    """Select configuration based on mode with CLI > env var > auto precedence."""
    # Detect system RAM
    ram_gb = _detect_ram_gb()
    
    # Determine mode with precedence: CLI > env var > auto
    mode = (manual_mode or os.getenv("RAG_MODE", "")).strip().lower()
    source = "cli" if manual_mode else "env" if os.getenv("RAG_MODE") else "auto"
    
    if not mode:
        mode = _auto_select_mode(ram_gb)
        source = "auto"
        logger.info(f"Auto-selected mode '{mode}' based on {ram_gb:.0f}GB detected RAM")

    legacy_mapping = {
        "high": "regular",
        "high-performance": "regular",
        "tier1": "regular",
        "power-fast": "regular",  # Deprecated: use 'regular' which auto-scales
        "efficiency": "regular",
        "tier2": "regular",
    }
    if mode in legacy_mapping:
        old_mode = mode
        mode = legacy_mapping[mode]
        logger.warning(f"Legacy tier '{old_mode}' mapped to mode '{mode}'. Use --mode={mode}")
    
    if mode not in VALID_MODES:
        valid_modes = ", ".join(sorted(VALID_MODES))
        raise ValueError(
            f"Unknown mode '{mode}'. Valid modes: {valid_modes}. "
            f"Set RAG_MODE env var or use --mode flag."
        )
    
    # Get RAM-aware configuration
    config = _get_mode_config(mode, ram_gb)
    
    logger.info(
        f"[Hardware: {ram_gb:.0f}GB] Selected mode: {mode} (source: {source}) | "
        f"LLM: {config.llm_model} | "
        f"Quantization: {config.quantization} | "
        f"Context: {config.context_window:,} | "
        f"Budget: {config.retrieval_budget:,}"
    )
    
    return config

