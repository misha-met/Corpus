#!/usr/bin/env bash
# ============================================================================
# run_all_tests.sh — Execute the full RAG test and analysis suite
#
# Usage:
#   cd <project-root>
#   bash tests/run_all_tests.sh
#
# Outputs:
#   tests/logs/       — Structured JSONL logs per test module
#   tests/results/    — JSON data files (e2e results, latency profiles)
#   stdout            — Test summary and pass/fail status
#
# Exit code:
#   0  — All tests passed
#   1  — One or more tests failed
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Ensure output directories exist
mkdir -p tests/logs tests/results

echo "============================================================"
echo "  RAG Pipeline Test & Analysis Suite"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Project: $PROJECT_ROOT"
echo "============================================================"
echo ""

# ------------------------------------------------------------------
# Phase 1: Unit tests (fast, deterministic)
# ------------------------------------------------------------------
echo ">>> Phase 1: Unit Tests"
echo "-----------------------------------------------------------"

UNIT_TESTS=(
    "tests/test_models_ingest.py"
    "tests/test_storage.py"
    "tests/test_budget_packing.py"
    "tests/test_citations_intent.py"
    "tests/test_config_metrics.py"
)

UNIT_FAILED=0
for test_file in "${UNIT_TESTS[@]}"; do
    echo ""
    echo "  Running: $test_file"
    if python -m pytest "$test_file" -v --tb=short -q 2>&1 | tail -20; then
        echo "  ✓ $test_file PASSED"
    else
        echo "  ✗ $test_file FAILED"
        UNIT_FAILED=1
    fi
done

echo ""
echo "-----------------------------------------------------------"
if [ $UNIT_FAILED -eq 0 ]; then
    echo ">>> Phase 1 Result: ALL UNIT TESTS PASSED"
else
    echo ">>> Phase 1 Result: SOME UNIT TESTS FAILED"
fi
echo ""

# ------------------------------------------------------------------
# Phase 2: Retrieval tests (component interaction)
# ------------------------------------------------------------------
echo ">>> Phase 2: Retrieval & Pipeline Tests"
echo "-----------------------------------------------------------"

RETRIEVAL_FAILED=0
echo "  Running: tests/test_retrieval.py"
if python -m pytest tests/test_retrieval.py -v --tb=short -q 2>&1 | tail -30; then
    echo "  ✓ Retrieval tests PASSED"
else
    echo "  ✗ Retrieval tests FAILED"
    RETRIEVAL_FAILED=1
fi

echo ""
echo "-----------------------------------------------------------"
echo ""

# ------------------------------------------------------------------
# Phase 3: Integration tests
# ------------------------------------------------------------------
echo ">>> Phase 3: Integration Tests"
echo "-----------------------------------------------------------"

INTEGRATION_FAILED=0
echo "  Running: tests/test_integration.py"
if python -m pytest tests/test_integration.py -v --tb=short -q 2>&1 | tail -30; then
    echo "  ✓ Integration tests PASSED"
else
    echo "  ✗ Integration tests FAILED"
    INTEGRATION_FAILED=1
fi

echo ""
echo "-----------------------------------------------------------"
echo ""

# ------------------------------------------------------------------
# Phase 4: End-to-end test
# ------------------------------------------------------------------
echo ">>> Phase 4: End-to-End Test"
echo "-----------------------------------------------------------"

E2E_FAILED=0
echo "  Running: tests/test_e2e.py"
if python -m pytest tests/test_e2e.py -v --tb=short -q 2>&1 | tail -30; then
    echo "  ✓ E2E tests PASSED"
else
    echo "  ✗ E2E tests FAILED"
    E2E_FAILED=1
fi

echo ""
echo "-----------------------------------------------------------"
echo ""

# ------------------------------------------------------------------
# Phase 5: Latency profiling
# ------------------------------------------------------------------
echo ">>> Phase 5: Latency Profiling"
echo "-----------------------------------------------------------"

LATENCY_FAILED=0
echo "  Running: tests/test_latency_profiler.py"
if python -m pytest tests/test_latency_profiler.py -v --tb=short -q 2>&1 | tail -40; then
    echo "  ✓ Latency profiler PASSED"
else
    echo "  ✗ Latency profiler FAILED"
    LATENCY_FAILED=1
fi

echo ""
echo "-----------------------------------------------------------"
echo ""

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
echo "============================================================"
echo "  FINAL SUMMARY"
echo "============================================================"
echo "  Unit Tests:        $([ $UNIT_FAILED -eq 0 ] && echo 'PASSED' || echo 'FAILED')"
echo "  Retrieval Tests:   $([ $RETRIEVAL_FAILED -eq 0 ] && echo 'PASSED' || echo 'FAILED')"
echo "  Integration Tests: $([ $INTEGRATION_FAILED -eq 0 ] && echo 'PASSED' || echo 'FAILED')"
echo "  E2E Tests:         $([ $E2E_FAILED -eq 0 ] && echo 'PASSED' || echo 'FAILED')"
echo "  Latency Profiling: $([ $LATENCY_FAILED -eq 0 ] && echo 'PASSED' || echo 'FAILED')"
echo ""
echo "  Logs:    tests/logs/"
echo "  Results: tests/results/"
echo "============================================================"

# Exit with failure if any phase failed
TOTAL_FAILED=$((UNIT_FAILED + RETRIEVAL_FAILED + INTEGRATION_FAILED + E2E_FAILED + LATENCY_FAILED))
if [ $TOTAL_FAILED -gt 0 ]; then
    echo ""
    echo "  ✗ $TOTAL_FAILED phase(s) had failures."
    exit 1
else
    echo ""
    echo "  ✓ All phases passed successfully."
    exit 0
fi
