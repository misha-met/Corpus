"""Mode-aware configuration for the RAG pipeline.

Provides model IDs, retrieval budgets, and per-intent generation parameters
for all supported hardware tiers.

Architecture
~~~~~~~~~~~~
- ``ModelConfig`` is a frozen dataclass holding a complete model stack and
  retrieval knobs for a given mode/RAM tier.  All downstream code reads from
  it; nothing mutates it after construction.
- ``select_mode_config()`` picks the right ``ModelConfig`` at engine startup
  via ``CLI > RAG_MODE env var > auto`` precedence.  RAM is detected once and
  cached.
- ``resolve_retrieval_params()`` applies per-intent scale factors on top of
  the base mode values so retrieval depth varies by query type without
  duplicating config tables.
- ``resolve_generation_params()`` returns per-intent sampling params (temp,
  top_p, thinking on/off).  Deep-research mode inherits regular params and
  overrides only the analytical intents.
"""
from __future__ import annotations

import logging
import os
import platform
from dataclasses import dataclass, replace as _dc_replace
from typing import Optional

logger = logging.getLogger(__name__)

CITATIONS_ENABLED_DEFAULT: bool = False
_detected_ram_gb: Optional[float] = None


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    val = raw.strip().lower()
    if val in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if val in {"0", "false", "f", "no", "n", "off"}:
        return False
    logger.warning("Invalid boolean for %s=%r; using default=%s", name, raw, default)
    return default


def _env_float(name: str, default: float, *, low: float | None = None, high: float | None = None) -> float:
    raw = os.getenv(name)
    if raw is None:
        value = default
    else:
        try:
            value = float(raw)
        except (TypeError, ValueError):
            logger.warning("Invalid float for %s=%r; using default=%s", name, raw, default)
            value = default

    if low is not None and value < low:
        logger.warning("%s=%s is below minimum %s; clamping", name, value, low)
        value = low
    if high is not None and value > high:
        logger.warning("%s=%s is above maximum %s; clamping", name, value, high)
        value = high
    return value


def _env_int(name: str, default: int, *, low: int | None = None, high: int | None = None) -> int:
    raw = os.getenv(name)
    if raw is None:
        value = default
    else:
        try:
            value = int(raw)
        except (TypeError, ValueError):
            logger.warning("Invalid int for %s=%r; using default=%s", name, raw, default)
            value = default

    if low is not None and value < low:
        logger.warning("%s=%s is below minimum %s; clamping", name, value, low)
        value = low
    if high is not None and value > high:
        logger.warning("%s=%s is above maximum %s; clamping", name, value, high)
        value = high
    return value


# Geotag and geocoder hardening knobs
GEOTAG_MIN_CONFIDENCE: float = _env_float("GEOTAG_MIN_CONFIDENCE", 0.5, low=0.0, high=1.0)
GEOTAG_FUZZY_THRESHOLD: int = _env_int("GEOTAG_FUZZY_THRESHOLD", 75, low=0, high=100)
GEOTAG_FUZZY_SCORE_FLOOR: int = _env_int("GEOTAG_FUZZY_SCORE_FLOOR", 78, low=0, high=100)
GEOTAG_FUZZY_MARGIN_THRESHOLD: float = _env_float("GEOTAG_FUZZY_MARGIN_THRESHOLD", 4.0, low=0.0)
GEOTAG_ENTITY_TYPE_PENALTY: float = _env_float("GEOTAG_ENTITY_TYPE_PENALTY", 0.12, low=0.0, high=1.0)
GEOTAG_GENERIC_TOKEN_PENALTY: float = _env_float("GEOTAG_GENERIC_TOKEN_PENALTY", 0.08, low=0.0, high=1.0)
GEOTAG_NER_CONTEXT_WINDOW: int = _env_int("GEOTAG_NER_CONTEXT_WINDOW", 8, low=0, high=48)
GEOTAG_NER_THRESHOLD: float = _env_float("GEOTAG_NER_THRESHOLD", 0.40, low=0.0, high=1.0)

# People dictionary ingest knobs
PEOPLETAG_MIN_CONFIDENCE: float = _env_float("PEOPLETAG_MIN_CONFIDENCE", 0.70, low=0.0, high=1.0)
PEOPLETAG_NER_CONTEXT_WINDOW: int = _env_int("PEOPLETAG_NER_CONTEXT_WINDOW", 8, low=0, high=48)
PEOPLETAG_NER_THRESHOLD: float = _env_float("PEOPLETAG_NER_THRESHOLD", 0.45, low=0.0, high=1.0)
PEOPLETAG_FUZZY_THRESHOLD_LASTNAME: int = _env_int("PEOPLETAG_FUZZY_THRESHOLD_LASTNAME", 96, low=0, high=100)
PEOPLETAG_FUZZY_THRESHOLD_FULLNAME: int = _env_int("PEOPLETAG_FUZZY_THRESHOLD_FULLNAME", 93, low=0, high=100)

# Rollback toggles
USE_HARDENED_GEOCODER: bool = _env_bool("USE_HARDENED_GEOCODER", False)
USE_SOURCE_IDS_FILTER: bool = _env_bool("USE_SOURCE_IDS_FILTER", True)


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for RAG pipeline models and retrieval parameters.

    Retrieval note: BM25/FTS uses LanceDB's default tokenizer for the indexed
    text column. If production relevance requires different lexical behavior,
    configure a custom LanceDB FTS tokenization profile when creating the
    index. Injecting HuggingFace tokenizers directly into LanceDB FTS is not
    currently supported.
    """
    mode: str
    llm_model: str
    embedding_model: str
    reranker_model: str
    summary_model: str = "mlx-community/LFM2-8B-A1B-4bit"
    embedding_device: str = "cpu"
    quantization: str = "4-bit"
    context_window: int = 128_000
    retrieval_budget: int = 100_000
    top_k_dense: int = 100
    top_k_sparse: int = 100
    top_k_fused: int = 50
    top_k_rerank: int = 30
    top_k_final: int = 5
    reranker_threshold: float = 0.05
    reranker_min_docs: int = 3
    bm25_weight: float = 0.5
    use_hybrid: bool = True
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
    top_k: int = 20
    min_p: float = 0.0
    presence_penalty: float = 1.5
    repetition_penalty: float = 1.0
    enable_thinking: bool = False


@dataclass(frozen=True)
class ResolvedRetrievalParams:
    top_k_dense: int
    top_k_fused: int
    top_k_rerank: int
    top_k_final: int
    reranker_threshold: float
    reranker_min_docs: int
    bm25_weight: float = 0.5
    use_hybrid: bool = True
    max_children_per_parent: int = 2


INTENT_RETRIEVAL_OVERRIDES: dict[str, IntentRetrievalOverrides] = {
    # FACTUAL: stricter threshold (1.3×) + min_docs=1 so a single high-confidence
    # fact is not padded with low-quality backfill chunks.
    "FACTUAL":    IntentRetrievalOverrides(top_k_dense_scale=0.85, top_k_fused_scale=0.5, top_k_final_scale=1.0, reranker_threshold_scale=1.3, reranker_min_docs=1),
    "SUMMARIZE":  IntentRetrievalOverrides(top_k_dense_scale=1.0, top_k_final_scale=1.0),
    "EXPLAIN":    IntentRetrievalOverrides(top_k_dense_scale=1.2, top_k_final_scale=1.2),
    "ANALYZE":    IntentRetrievalOverrides(top_k_dense_scale=1.3, top_k_final_scale=1.5, reranker_threshold_scale=0.8),
    "COMPARE":    IntentRetrievalOverrides(top_k_dense_scale=1.3, top_k_final_scale=1.5, reranker_threshold_scale=0.8),
    "CRITIQUE":   IntentRetrievalOverrides(top_k_dense_scale=1.2, top_k_final_scale=1.25, reranker_threshold_scale=0.85),
    "COLLECTION": IntentRetrievalOverrides(top_k_dense_scale=1.1, top_k_fused_scale=0.5, top_k_final_scale=1.0, reranker_threshold_scale=1.2),
    "EXTRACT":    IntentRetrievalOverrides(top_k_dense_scale=0.9, top_k_final_scale=1.0, reranker_threshold_scale=1.2, reranker_min_docs=1),
    "TIMELINE":   IntentRetrievalOverrides(top_k_dense_scale=1.4, top_k_fused_scale=1.2, top_k_final_scale=1.5, reranker_threshold_scale=0.8),
    "HOW_TO":     IntentRetrievalOverrides(top_k_dense_scale=1.1, top_k_final_scale=1.2),
    "QUOTE_EVIDENCE": IntentRetrievalOverrides(top_k_dense_scale=0.9, top_k_fused_scale=0.7, top_k_final_scale=0.8, reranker_threshold_scale=1.35, reranker_min_docs=1),
}

INTENT_GENERATION_PARAMS_REGULAR: dict[str, IntentGenerationParams] = {
    "FACTUAL":        IntentGenerationParams(temperature=0.5, top_p=0.7, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0, enable_thinking=False),
    "SUMMARIZE":      IntentGenerationParams(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0, enable_thinking=False),
    "EXPLAIN":        IntentGenerationParams(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0, enable_thinking=False),
    "ANALYZE":        IntentGenerationParams(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0, enable_thinking=False),
    "COMPARE":        IntentGenerationParams(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0, enable_thinking=False),
    "CRITIQUE":       IntentGenerationParams(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0, enable_thinking=False),
    "COLLECTION":     IntentGenerationParams(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0, enable_thinking=False),
    "OVERVIEW":       IntentGenerationParams(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0, enable_thinking=False),
    "EXTRACT":        IntentGenerationParams(temperature=0.5, top_p=0.7, top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0, enable_thinking=False),
    "TIMELINE":       IntentGenerationParams(temperature=0.5, top_p=0.7, top_k=20, min_p=0.0, presence_penalty=1.0, repetition_penalty=1.0, enable_thinking=False),
    "HOW_TO":         IntentGenerationParams(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0, enable_thinking=False),
    "QUOTE_EVIDENCE": IntentGenerationParams(temperature=0.3, top_p=0.6, top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0, enable_thinking=False),
}

INTENT_GENERATION_PARAMS_DEEP_RESEARCH: dict[str, IntentGenerationParams] = {
    **INTENT_GENERATION_PARAMS_REGULAR,
    # Deep research overrides: higher temperature + thinking enabled
    "EXPLAIN":  _dc_replace(INTENT_GENERATION_PARAMS_REGULAR["EXPLAIN"],  temperature=1.0, top_p=0.95, enable_thinking=True),
    "ANALYZE":  _dc_replace(INTENT_GENERATION_PARAMS_REGULAR["ANALYZE"],  temperature=1.0, top_p=0.95, enable_thinking=True),
    "COMPARE":  _dc_replace(INTENT_GENERATION_PARAMS_REGULAR["COMPARE"],  temperature=1.0, top_p=0.95, enable_thinking=True),
    "CRITIQUE": _dc_replace(INTENT_GENERATION_PARAMS_REGULAR["CRITIQUE"], temperature=1.0, top_p=0.95, enable_thinking=True),
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
        bm25_weight=mode_config.bm25_weight,
        use_hybrid=mode_config.use_hybrid,
        max_children_per_parent=mode_config.max_children_per_parent,
    )


def resolve_generation_params(intent: str, mode: str = "regular") -> IntentGenerationParams:
    """Return generation params for intent and mode. Falls back to OVERVIEW if unrecognized."""
    if mode == "deep-research":
        params = INTENT_GENERATION_PARAMS_DEEP_RESEARCH.get(intent.upper())
        if params is not None:
            return params
    params = INTENT_GENERATION_PARAMS_REGULAR.get(intent.upper())
    if params is not None:
        return params
    return INTENT_GENERATION_PARAMS_REGULAR.get("OVERVIEW", IntentGenerationParams(
        temperature=0.7, top_p=0.8, enable_thinking=False,
    ))


def _get_mode_config(mode: str, ram_gb: float) -> ModelConfig:
    """Get mode configuration with RAM-aware token budget adjustments."""

    # Shared model stack per mode (fields identical across RAM tiers)
    _REGULAR_MODELS = dict(
        llm_model="NexVeridian/Qwen3.5-35B-A3B-4bit",
        summary_model="mlx-community/LFM2-8B-A1B-4bit",
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
                top_k_rerank=55,
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

    elif mode == "deep-research":
        if ram_gb < 48:
            logger.warning(
                "deep-research mode requires 48GB+ RAM. Detected %.0fGB. "
                "Falling back to regular mode.",
                ram_gb,
            )
            return _get_mode_config("regular", ram_gb)
        return ModelConfig(
            mode="deep-research",
            **_REGULAR_MODELS,
            context_window=64_000,
            retrieval_budget=40_000,
            top_k_dense=400,
            top_k_sparse=400,
            top_k_fused=200,
            top_k_rerank=80,
            top_k_final=8,
            reranker_threshold=0.04,
            reranker_min_docs=4,
            reranker_enabled=True,
            context_expansion_enabled=True,
            system_ram_gb=ram_gb,
        )

    raise ValueError(f"Unknown mode: {mode}")


VALID_MODES = {"regular", "deep-research"}
MODE_RAM_REQUIREMENTS: dict[str, float] = {
    "regular": 32.0,
    "deep-research": 48.0,
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


def select_mode_config(*, manual_mode: Optional[str] = None) -> ModelConfig:
    """Select configuration based on mode with CLI > env var > auto precedence."""
    # Detect system RAM
    ram_gb = _detect_ram_gb()
    
    # Determine mode with precedence: CLI > env var > auto
    mode = (manual_mode or os.getenv("RAG_MODE", "")).strip().lower()
    source = "cli" if manual_mode else "env" if os.getenv("RAG_MODE") else "auto"
    
    if not mode:
        mode = "regular"
        source = "auto"
        logger.info(f"Auto-selected mode '{mode}' based on {ram_gb:.0f}GB detected RAM")

    legacy_mapping = {
        "high": "regular",
        "high-performance": "regular",
        "tier1": "regular",
        "power-fast": "regular",  # Deprecated: use 'regular' which auto-scales
        "efficiency": "regular",
        "tier2": "regular",
        "power-deep-research": "deep-research",  # Legacy: consolidated to single model
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
    
    config = _get_mode_config(mode, ram_gb)
    
    logger.info(
        f"[Hardware: {ram_gb:.0f}GB] Selected mode: {mode} (source: {source}) | "
        f"LLM: {config.llm_model} | "
        f"Quantization: {config.quantization} | "
        f"Context: {config.context_window:,} | "
        f"Budget: {config.retrieval_budget:,}"
    )
    
    return config

