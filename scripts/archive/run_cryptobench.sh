#!/bin/bash
# ============================================================================
# PRISM-LBS BENCHMARK — LOCAL DATA ONLY
# This script assumes all datasets are pre-placed in benchmarks/datasets/
# No auto-download. No network access. Ever.
# ============================================================================
#
# CryptoBench External Ground Truth Benchmark Runner
#
# REQUIRED DATA (must be manually provided):
#   - benchmarks/datasets/cryptobench/dataset.json
#   - benchmarks/datasets/cryptobench/folds.json
#   - benchmarks/datasets/cryptobench/structures/test/*.cif or *.pdb
#
# Download sources (manual only):
#   - Dataset: https://osf.io/pz4a9/
#   - Structures: https://files.rcsb.org/download/{pdb_id}.cif
#
# Usage:
#   ./scripts/run_cryptobench.sh [--quick]
#
# Options:
#   --quick              Run on 20 structures for quick testing
#   --allow-non-test     REQUIRED flag to use non-test splits (disabled by default)
#   --split SPLIT        Specify split (only with --allow-non-test)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Paths
CRYPTOBENCH_DIR="$PROJECT_ROOT/benchmarks/datasets/cryptobench"
PRISM_BIN="$PROJECT_ROOT/target/release/prism-lbs"
EVAL_SCRIPT="$SCRIPT_DIR/cryptobench_evaluate.py"

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

# Parse arguments
QUICK_MODE=false
SPLIT="test"
ALLOW_NON_TEST=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick) QUICK_MODE=true; shift ;;
        --allow-non-test) ALLOW_NON_TEST=true; shift ;;
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --download)
            echo -e "${RED}ERROR: --download flag is not supported.${NC}"
            echo "This script does NOT auto-download data."
            echo "Please manually download structures to: $CRYPTOBENCH_DIR/structures/$SPLIT/"
            echo "Source: https://files.rcsb.org/download/{pdb_id}.cif"
            exit 1
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ============================================================================
# ENFORCE TEST-SPLIT-ONLY BY DEFAULT
# ============================================================================
if [[ "$SPLIT" != "test" ]] && [[ "$ALLOW_NON_TEST" == "false" ]]; then
    echo -e "${RED}ERROR: Non-test splits are disabled by default.${NC}"
    echo ""
    echo "To use split '$SPLIT', you MUST pass --allow-non-test explicitly:"
    echo "  ./scripts/run_cryptobench.sh --allow-non-test --split $SPLIT"
    echo ""
    echo "Default behavior uses only the official test split."
    exit 1
fi

echo ""
echo "============================================================"
echo "   CryptoBench External Ground Truth Benchmark"
echo "   (NO AUTO-DOWNLOADS - Manual data provision required)"
echo "============================================================"
echo ""

# ============================================================================
# MANDATORY DATA CHECKS - Exit with error if missing
# ============================================================================

# Check for dataset.json
if [[ ! -f "$CRYPTOBENCH_DIR/dataset.json" ]]; then
    echo -e "${RED}ERROR: Missing required benchmark dataset: CryptoBench dataset.json${NC}"
    echo ""
    echo "Please place it in: $CRYPTOBENCH_DIR/dataset.json"
    echo ""
    echo "Download from: https://osf.io/pz4a9/"
    echo "Direct link:   curl -sL 'https://osf.io/download/ta2ju/' -o $CRYPTOBENCH_DIR/dataset.json"
    exit 1
fi

# Check for folds.json
if [[ ! -f "$CRYPTOBENCH_DIR/folds.json" ]]; then
    echo -e "${RED}ERROR: Missing required benchmark dataset: CryptoBench folds.json${NC}"
    echo ""
    echo "Please place it in: $CRYPTOBENCH_DIR/folds.json"
    echo ""
    echo "Download from: https://osf.io/pz4a9/"
    echo "Direct link:   curl -sL 'https://osf.io/download/5s93p/' -o $CRYPTOBENCH_DIR/folds.json"
    exit 1
fi

# Check for structures directory
STRUCTURES_DIR="$CRYPTOBENCH_DIR/structures/$SPLIT"
if [[ ! -d "$STRUCTURES_DIR" ]]; then
    echo -e "${RED}ERROR: Missing required structures directory${NC}"
    echo ""
    echo "Please create and populate: $STRUCTURES_DIR/"
    echo ""
    echo "Download structures from RCSB:"
    echo "  https://files.rcsb.org/download/{pdb_id}.cif"
    exit 1
fi

# Check if structures exist
N_STRUCTURES=$(ls "$STRUCTURES_DIR"/*.{cif,pdb} 2>/dev/null | wc -l || echo "0")
if [[ "$N_STRUCTURES" -eq 0 ]]; then
    echo -e "${RED}ERROR: No structure files found in $STRUCTURES_DIR${NC}"
    echo ""
    echo "Please download CIF/PDB files to this directory."
    echo "Source: https://files.rcsb.org/download/{pdb_id}.cif"
    exit 1
fi

# Check prism binary
if [[ ! -f "$PRISM_BIN" ]]; then
    echo -e "${RED}ERROR: prism-lbs binary not found${NC}"
    echo ""
    echo "Please build with: cargo build --release --features cuda -p prism-lbs"
    exit 1
fi

echo -e "${GREEN}All required data present${NC}"
echo ""

# Get test set PDB IDs
echo "Split: $SPLIT"
TEST_PDBS=$(python3 -c "
import json
with open('$CRYPTOBENCH_DIR/folds.json') as f:
    folds = json.load(f)
pdbs = folds.get('$SPLIT', [])
print(' '.join(pdbs))
")
N_TOTAL=$(echo $TEST_PDBS | wc -w)
echo "Structures in split: $N_TOTAL"
echo "Structures available: $N_STRUCTURES"

# Quick mode - only 20 structures
if $QUICK_MODE; then
    TEST_PDBS=$(echo $TEST_PDBS | tr ' ' '\n' | head -20 | tr '\n' ' ')
    N_TOTAL=20
    echo -e "${YELLOW}Quick mode: using first 20 structures${NC}"
fi

# Setup directories
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PREDICTIONS_DIR="$CRYPTOBENCH_DIR/predictions/${SPLIT}_${TIMESTAMP}"
RESULTS_FILE="$CRYPTOBENCH_DIR/results_${SPLIT}_${TIMESTAMP}.json"

mkdir -p "$PREDICTIONS_DIR"

# Run PRISM predictions
echo ""
echo -e "${GREEN}Running PRISM-LBS predictions...${NC}"
START_TIME=$(date +%s)

"$PRISM_BIN" \
    -i "$STRUCTURES_DIR" \
    -o "$PREDICTIONS_DIR" \
    --format json \
    --publication \
    --gpu-geometry \
    --unified \
    batch --parallel 4

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

N_PREDICTIONS=$(ls "$PREDICTIONS_DIR"/*.json 2>/dev/null | wc -l)
echo ""
echo -e "${GREEN}Generated $N_PREDICTIONS predictions in ${DURATION}s${NC}"

# Run evaluation
echo ""
echo -e "${GREEN}Evaluating against CryptoBench ground truth...${NC}"
echo ""

python3 "$EVAL_SCRIPT" \
    --predictions "$PREDICTIONS_DIR" \
    --split "$SPLIT" \
    --output "$RESULTS_FILE"

echo ""
echo "============================================================"
echo "   Benchmark Complete"
echo "============================================================"
echo ""
echo "Results saved to: $RESULTS_FILE"
echo "Predictions in: $PREDICTIONS_DIR"
echo ""

# Summary
echo -e "${BLUE}External Ground Truth Source:${NC}"
echo "  Dataset: CryptoBench (Bioinformatics 2024)"
echo "  OSF: https://osf.io/pz4a9/"
echo "  GitHub: https://github.com/skrhakv/CryptoBench"
echo ""
