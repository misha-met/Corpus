# RAG Pipeline Test Suite

Test and analysis suite for the offline RAG pipeline, optimised for Apple Silicon.

## Hardware Baseline

Results in this repository were generated on:

| Property | Value |
|----------|-------|
| Chip | Apple M1 Pro |
| CPU cores | 8 (6 performance + 2 efficiency) |
| GPU cores | 14-core Apple GPU |
| RAM | 32 GB unified memory |
| OS | macOS 26.2 |
| Python | 3.13.7 |
| Architecture | arm64 |

These results serve as a **baseline** for comparison against future hardware
(e.g. Apple M4 Max 64 GB). All experiment result files include hardware
metadata so comparisons can be validated programmatically.

## What the test suite covers

### Baseline tests (223 tests)

| Module | Tests | Coverage |
|--------|------:|----------|
| `test_models_ingest.py` | 45 | Data models, tokenisation, chunking, immutability |
| `test_storage.py` | 22 | SQLite, Chroma, BM25, persist/reload |
| `test_retrieval.py` | 41 | Dense/sparse search, RRF fusion, dedup, rerank, thresholds |
| `test_budget_packing.py` | 23 | Token budget, truncation, metadata alignment |
| `test_citations_intent.py` | 32 | Citation formatting, intent classification |
| `test_config_metrics.py` | 25 | Mode configs, auto-selection, metrics |
| `test_integration.py` | 14 | Cross-stage composition |
| `test_e2e.py` | 3 | Full pipeline (30 runs), determinism |
| `test_latency_profiler.py` | 10 | Per-stage latency profiles |

### Optimisation experiments (13 tests)

| Experiment | Tests | What it measures |
|-----------|------:|------------------|
| **Exp 1: Vector warmup** | 2 | Cold-start vs warm-start latency |
| **Exp 2: Dedup ordering** | 2 | Dedup-before vs dedup-after reranking |
| **Exp 3: Real models** | 4 | Embedding encode, reranker batch scaling, pipeline, memory (skips if models unavailable) |
| **Exp 4: Caching** | 2 | Query embedding cache hit rate and latency |
| **Exp 5: Budget utilisation** | 3 | Token budget fill rate, truncation, document sizes |

**Total: 236 tests, 0 failures**

## How to run

### Prerequisites

```bash
# From project root
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pytest
```

### Run all tests

```bash
# Full suite (baseline + experiments)
.venv/bin/python -m pytest tests/ -v

# Or use the runner script (5-phase execution)
./tests/run_all_tests.sh
```

### Run specific modules

```bash
# Baseline only
.venv/bin/python -m pytest tests/ -k "not Exp" -v

# Experiments only
.venv/bin/python -m pytest tests/test_experiments.py -v

# Mock-only experiments (no model downloads needed)
.venv/bin/python -m pytest tests/test_experiments.py -k "not Exp3" -v

# Real model experiments (requires BAAI/bge-m3 and Jina reranker)
.venv/bin/python -m pytest tests/test_experiments.py -k "Exp3" -v
```

### Run with timing details

```bash
.venv/bin/python -m pytest tests/ -v --durations=20
```

## Artifacts produced

After a full run, the following files are generated:

### Results (`tests/results/`)

| File | Contents |
|------|----------|
| `e2e_results.json` | 30 structured E2E pipeline results |
| `latency_profiles.json` | Per-query pipeline latency profiles |
| `latency_summary.json` | Per-stage latency breakdown |
| `exp1_warmup.json` | Warmup experiment results |
| `exp2_dedup_order.json` | Dedup ordering experiment |
| `exp3_embedding_latency.json` | Real embedding model latency |
| `exp3_reranker_batch_scaling.json` | Reranker batch size scaling |
| `exp3_real_pipeline.json` | Full pipeline with real embeddings |
| `exp3_memory.json` | Memory usage profiling |
| `exp4_caching.json` | Embedding cache experiment |
| `exp5_budget_utilisation.json` | Budget fill analysis by mode |
| `exp5_truncation.json` | Truncation impact analysis |
| `exp5_doc_sizes.json` | Document size distribution |

### Logs (`tests/logs/`)

Structured JSON-lines logs per module per run, timestamped. Each entry includes
timing data, context, and log level for post-hoc analysis.

### Report

`tests/TEST_REPORT.md` — the consolidated analysis report with baseline
measurements, experiment results, and optimisation recommendations.

## Comparing results across hardware

All result JSON files include a `hardware` object with chip, core counts, RAM,
OS version, and Python version. To compare results from a different machine:

1. Run the full suite on the new hardware
2. Load both sets of JSON results
3. Compare by matching experiment name and normalising by hardware metadata

Example comparison script:

```python
import json

with open("tests/results/exp3_embedding_latency.json") as f:
    data = json.load(f)

print(f"Chip: {data['hardware']['chip']}")
print(f"Single encode: {data['single_text']['mean_ms']:.2f} ms")
print(f"Batch (5): {data['batch_5_texts']['mean_ms']:.2f} ms")
```

The hardware detection module (`tests/hardware_info.py`) auto-detects Apple
Silicon chip, core counts, and RAM via `sysctl`. It can be imported directly:

```python
from tests.hardware_info import HARDWARE
print(HARDWARE.summary())
```

## Design notes

- **No changes to source code.** All tests live in `tests/` and use mocks or
  real models loaded independently.
- **Deterministic mocks.** Hash-based embeddings, word-overlap reranker, and
  whitespace tokeniser ensure reproducible results without GPU.
- **Conditional real-model tests.** Experiment 3 tests auto-skip if models
  are not downloaded, so CI can run mock tests without hardware requirements.
- **Fixed corpus and queries.** The same 5 documents and 5 queries are used
  across all experiments for direct comparability.
