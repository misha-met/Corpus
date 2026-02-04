from __future__ import annotations

import logging
import os
import platform
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for RAG pipeline models and retrieval parameters.
    
    Attributes:
        mode: Operating mode (regular, power-fast, power-deep-research)
        llm_model: MLX model identifier for generation
        embedding_model: Sentence transformer model for embeddings
        reranker_model: FlagEmbedding reranker model
        embedding_device: Device for embedding model (cpu to preserve VRAM)
        quantization: MLX quantization level (4-bit, 8-bit)
        context_window: Maximum context window size in tokens
        retrieval_budget: Token budget for retrieved context
        top_k_dense: Number of dense search results
        top_k_sparse: Number of sparse (BM25) search results
        top_k_fused: Number of results after RRF fusion
        top_k_rerank: Number of results to pass to reranker
        top_k_final: Number of final results to return
    """
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


# =============================================================================
# Mode Preset Configurations
# =============================================================================

# Regular mode: Balanced performance with 4-bit quantization
# Suitable for most queries with moderate retrieval depth
REGULAR_CONFIG = ModelConfig(
    mode="regular",
    llm_model="mlx-community/Qwen3-30B-A3B-Instruct-4bit",
    embedding_model="BAAI/bge-m3",
    reranker_model="BAAI/bge-reranker-v2-m3",
    embedding_device="cpu",
    quantization="4-bit",
    context_window=128_000,
    retrieval_budget=100_000,
    top_k_dense=100,
    top_k_sparse=100,
    top_k_fused=50,
    top_k_rerank=20,
    top_k_final=5,
)

# Power-fast mode: Higher precision with 8-bit quantization
# Deeper retrieval for comprehensive answers with faster model
POWER_FAST_CONFIG = ModelConfig(
    mode="power-fast",
    llm_model="mlx-community/Qwen3-30B-A3B-Instruct-2507-8bit",
    embedding_model="BAAI/bge-m3",
    reranker_model="BAAI/bge-reranker-v2-m3",
    embedding_device="cpu",
    quantization="8-bit",
    context_window=128_000,
    retrieval_budget=100_000,
    top_k_dense=300,
    top_k_sparse=300,
    top_k_fused=150,
    top_k_rerank=80,
    top_k_final=15,
)

# Power-deep-research mode: Larger model with constrained context
# Best for complex research queries requiring deeper reasoning
POWER_DEEP_RESEARCH_CONFIG = ModelConfig(
    mode="power-deep-research",
    llm_model="mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit",
    embedding_model="BAAI/bge-m3",
    reranker_model="BAAI/bge-reranker-v2-m3",
    embedding_device="cpu",
    quantization="4-bit",
    context_window=32_000,
    retrieval_budget=20_000,
    top_k_dense=300,
    top_k_sparse=300,
    top_k_fused=150,
    top_k_rerank=80,
    top_k_final=15,
)

# Mode lookup table
MODE_CONFIGS: dict[str, ModelConfig] = {
    "regular": REGULAR_CONFIG,
    "power-fast": POWER_FAST_CONFIG,
    "power-deep-research": POWER_DEEP_RESEARCH_CONFIG,
}

# RAM requirements per mode (in GB)
MODE_RAM_REQUIREMENTS: dict[str, float] = {
    "regular": 32.0,
    "power-fast": 48.0,
    "power-deep-research": 96.0,
}


# =============================================================================
# System Detection
# =============================================================================

def _detect_ram_gb() -> float:
    """Detect system RAM in gigabytes.
    
    Uses psutil if available for cross-platform accuracy,
    falls back to platform-specific methods.
    
    Returns:
        System RAM in GB, or 0.0 if detection fails.
    """
    # Try psutil first (most reliable)
    try:
        import psutil
        return psutil.virtual_memory().total / (1024**3)
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
            return int(output.strip()) / (1024**3)
        except Exception as exc:
            logger.debug(f"macOS RAM detection failed: {exc}")
            return 0.0

    # Fallback: Linux sysconf
    if hasattr(os, "sysconf"):
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return (pages * page_size) / (1024**3)
        except (ValueError, OSError, AttributeError) as exc:
            logger.debug(f"sysconf RAM detection failed: {exc}")
            return 0.0

    return 0.0


def _validate_ram_for_mode(mode: str, ram_gb: float) -> tuple[bool, str]:
    """Validate if system RAM meets requirements for specified mode.
    
    Args:
        mode: The target mode to validate
        ram_gb: Detected system RAM in GB
        
    Returns:
        Tuple of (is_valid, warning_message)
    """
    required = MODE_RAM_REQUIREMENTS.get(mode, 32.0)
    if ram_gb <= 0:
        return True, ""  # Can't validate, assume OK
    
    if ram_gb < required:
        return False, (
            f"Warning: Mode '{mode}' recommends {required:.0f}GB RAM, "
            f"but only {ram_gb:.1f}GB detected. Performance may be degraded."
        )
    return True, ""


def _auto_select_mode(ram_gb: float) -> str:
    """Auto-select the best mode based on available RAM.
    
    Args:
        ram_gb: Detected system RAM in GB
        
    Returns:
        Recommended mode name
    """
    if ram_gb >= 96:
        return "power-deep-research"
    elif ram_gb >= 48:
        return "power-fast"
    else:
        return "regular"


# =============================================================================
# Mode Selection
# =============================================================================

def select_mode_config(
    *,
    manual_mode: Optional[str] = None,
    validate_ram: bool = True,
) -> ModelConfig:
    """Select configuration based on mode with CLI/env var precedence.
    
    Resolution order:
    1. manual_mode argument (from CLI --mode flag)
    2. RAG_MODE environment variable
    3. Auto-detection based on system RAM
    
    Args:
        manual_mode: Mode override from CLI (highest priority)
        validate_ram: Whether to check RAM meets mode requirements
        
    Returns:
        ModelConfig for the selected mode
        
    Raises:
        ValueError: If specified mode is unknown
    """
    # Determine mode with precedence: CLI > env var > auto
    mode = (manual_mode or os.getenv("RAG_MODE", "")).strip().lower()
    source = "cli" if manual_mode else "env" if os.getenv("RAG_MODE") else "auto"
    
    ram_gb = _detect_ram_gb()
    
    if not mode:
        mode = _auto_select_mode(ram_gb)
        source = "auto"
        logger.info(
            f"Auto-selected mode '{mode}' based on {ram_gb:.1f}GB detected RAM"
        )
    
    # Handle legacy tier names for backward compatibility
    legacy_mapping = {
        "high": "power-fast",
        "high-performance": "power-fast",
        "tier1": "power-fast",
        "efficiency": "regular",
        "tier2": "regular",
    }
    if mode in legacy_mapping:
        old_mode = mode
        mode = legacy_mapping[mode]
        logger.warning(
            f"Legacy tier '{old_mode}' mapped to mode '{mode}'. "
            f"Please update to use --mode={mode}"
        )
    
    if mode not in MODE_CONFIGS:
        valid_modes = ", ".join(sorted(MODE_CONFIGS.keys()))
        raise ValueError(
            f"Unknown mode '{mode}'. Valid modes: {valid_modes}. "
            f"Set RAG_MODE env var or use --mode flag."
        )
    
    # Validate RAM if requested
    if validate_ram and ram_gb > 0:
        is_valid, warning = _validate_ram_for_mode(mode, ram_gb)
        if not is_valid:
            logger.warning(warning)
    
    config = MODE_CONFIGS[mode]
    logger.info(
        f"Selected mode: {mode} (source: {source}) | "
        f"LLM: {config.llm_model} | "
        f"Quantization: {config.quantization} | "
        f"Context: {config.context_window:,} | "
        f"Budget: {config.retrieval_budget:,}"
    )
    
    return config


# =============================================================================
# Legacy API (Deprecated)
# =============================================================================

def select_model_config(*, manual_tier: Optional[str] = None) -> ModelConfig:
    """Legacy function for backward compatibility.
    
    Deprecated: Use select_mode_config() instead.
    
    Args:
        manual_tier: Legacy tier name
        
    Returns:
        ModelConfig from the mapped mode
    """
    logger.warning(
        "select_model_config() is deprecated. Use select_mode_config() instead."
    )
    return select_mode_config(manual_mode=manual_tier)
