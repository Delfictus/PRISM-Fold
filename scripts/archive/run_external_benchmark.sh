#!/bin/bash
#
# PRISM-LBS External Ground Truth Benchmark Runner
# Uses LIGYSIS-style Top-N+K recall evaluation with external ground truth:
#   - Tier 1: 10 PDB co-crystal structures (classic drug binding sites)
#   - CryptoSite: 18 structures (cryptic binding site benchmark)
#   - ASBench: 15 structures (allosteric site benchmark)
#   - Novel: 10 structures (historically undruggable targets)
#
# Total: 53 structures with validated external ground truth
#
# Usage:
#   ./scripts/run_external_benchmark.sh [--regenerate]
#
# Options:
#   --regenerate   Re-run PRISM predictions (default: use cached)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Paths
STRUCTURES_DIR="$PROJECT_ROOT/benchmark/complete_validation/structures"
RESULTS_DIR="$PROJECT_ROOT/benchmark/complete_validation/results"
PRISM_BIN="$PROJECT_ROOT/target/release/prism-lbs"
EVAL_SCRIPT="$SCRIPT_DIR/external_benchmark_evaluate.py"

# Environment
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.6}"
export PRISM_PTX_DIR="${PRISM_PTX_DIR:-$PROJECT_ROOT/target/ptx}"
export RUST_LOG="${RUST_LOG:-warn}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo ""
echo "============================================================"
echo "   PRISM-LBS External Ground Truth Benchmark"
echo "   (LIGYSIS-style Top-N+K Recall Evaluation)"
echo "============================================================"
echo ""

REGENERATE=false
if [[ "$1" == "--regenerate" ]]; then
    REGENERATE=true
    echo -e "${YELLOW}Regenerating all predictions...${NC}"
fi

# Check prerequisites
if [[ ! -f "$PRISM_BIN" ]]; then
    echo -e "${RED}Error: prism-lbs binary not found at $PRISM_BIN${NC}"
    echo "Run: cargo build --release --features cuda -p prism-lbs"
    exit 1
fi

# Create temp directory for consolidated predictions
TEMP_PREDS=$(mktemp -d)
trap "rm -rf $TEMP_PREDS" EXIT

# Run predictions if regenerating
if $REGENERATE; then
    for tier in tier1 tier2_cryptosite tier2_asbench tier3_novel; do
        if [[ -d "$STRUCTURES_DIR/$tier" ]]; then
            echo -e "${BLUE}Processing $tier...${NC}"
            mkdir -p "$RESULTS_DIR/$tier"

            "$PRISM_BIN" \
                -i "$STRUCTURES_DIR/$tier" \
                -o "$RESULTS_DIR/$tier" \
                --format json \
                --publication \
                --gpu-geometry \
                --unified \
                batch --parallel 4
        fi
    done
fi

# Consolidate results
echo ""
echo "Consolidating predictions..."
for tier in tier1 tier2_cryptosite tier2_asbench tier3_novel; do
    if [[ -d "$RESULTS_DIR/$tier" ]]; then
        cp -f "$RESULTS_DIR/$tier"/*.json "$TEMP_PREDS/" 2>/dev/null || true
    fi
done

COUNT=$(ls "$TEMP_PREDS"/*.json 2>/dev/null | wc -l)
echo "Found $COUNT prediction files"

# Run evaluation
echo ""
echo -e "${GREEN}Running LIGYSIS-style evaluation...${NC}"
echo ""

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_CSV="$PROJECT_ROOT/benchmark/complete_validation/external_benchmark_$TIMESTAMP.csv"

python3 "$EVAL_SCRIPT" \
    --json-dir "$TEMP_PREDS" \
    --output "$OUTPUT_CSV" \
    --verbose

echo ""
echo "============================================================"
echo "   Benchmark Complete"
echo "============================================================"
echo ""
echo "Results saved to: $OUTPUT_CSV"
echo ""

# Summary
echo -e "${BLUE}Ground Truth Sources:${NC}"
echo "  - PDB co-crystal structures (drug binding sites)"
echo "  - CryptoSite paper: Cimermancic et al. J Mol Biol 2016"
echo "  - ASBench: Huang et al. Nucleic Acids Res 2015"
echo "  - Novel targets: FDA-approved drug binding sites"
echo ""
