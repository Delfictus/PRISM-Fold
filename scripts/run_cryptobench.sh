#!/bin/bash
#
# CryptoBench External Ground Truth Benchmark Runner
#
# Downloads and evaluates PRISM-LBS against CryptoBench:
#   - 1,107 APO structures with cryptic binding sites
#   - 5,493 ground truth annotations from holo structures
#   - Official train/test splits
#
# External Source: https://osf.io/pz4a9/
# Paper: Bioinformatics 2024, btae745
#
# Usage:
#   ./scripts/run_cryptobench.sh [--quick] [--download]
#
# Options:
#   --quick      Run on 20 structures for quick testing
#   --download   Download PDB structures from RCSB
#   --split      Specify split (test, train-0, train-1, train-2, train-3)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Paths
CRYPTOBENCH_DIR="$PROJECT_ROOT/benchmark/cryptobench"
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
DOWNLOAD_MODE=false
SPLIT="test"
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick) QUICK_MODE=true; shift ;;
        --download) DOWNLOAD_MODE=true; shift ;;
        --split) SPLIT="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo ""
echo "============================================================"
echo "   CryptoBench External Ground Truth Benchmark"
echo "   (Source: https://osf.io/pz4a9/)"
echo "============================================================"
echo ""

# Check for dataset
if [[ ! -f "$CRYPTOBENCH_DIR/dataset.json" ]]; then
    echo -e "${YELLOW}Downloading CryptoBench dataset from OSF...${NC}"
    mkdir -p "$CRYPTOBENCH_DIR"
    curl -sL "https://osf.io/download/ta2ju/" -o "$CRYPTOBENCH_DIR/dataset.json"
    curl -sL "https://osf.io/download/5s93p/" -o "$CRYPTOBENCH_DIR/folds.json"
    curl -sL "https://osf.io/download/x9dce/" -o "$CRYPTOBENCH_DIR/README.md"
    echo -e "${GREEN}Downloaded CryptoBench dataset (8MB)${NC}"
fi

# Get test set PDB IDs
echo ""
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

# Quick mode - only 20 structures
if $QUICK_MODE; then
    TEST_PDBS=$(echo $TEST_PDBS | tr ' ' '\n' | head -20 | tr '\n' ' ')
    N_TOTAL=20
    echo -e "${YELLOW}Quick mode: using first 20 structures${NC}"
fi

# Setup directories
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
STRUCTURES_DIR="$CRYPTOBENCH_DIR/structures/$SPLIT"
PREDICTIONS_DIR="$CRYPTOBENCH_DIR/predictions/${SPLIT}_${TIMESTAMP}"
RESULTS_FILE="$CRYPTOBENCH_DIR/results_${SPLIT}_${TIMESTAMP}.json"

mkdir -p "$STRUCTURES_DIR"
mkdir -p "$PREDICTIONS_DIR"

# Download structures if needed
if $DOWNLOAD_MODE; then
    echo ""
    echo -e "${BLUE}Downloading PDB structures from RCSB...${NC}"

    for pdb_id in $TEST_PDBS; do
        pdb_lower=$(echo "$pdb_id" | tr '[:upper:]' '[:lower:]')

        # Try CIF first, then PDB
        if [[ ! -f "$STRUCTURES_DIR/${pdb_lower}.cif" ]] && [[ ! -f "$STRUCTURES_DIR/${pdb_lower}.pdb" ]]; then
            if curl -sf "https://files.rcsb.org/download/${pdb_lower}.cif" -o "$STRUCTURES_DIR/${pdb_lower}.cif" 2>/dev/null; then
                echo -n "."
            elif curl -sf "https://files.rcsb.org/download/${pdb_lower}.pdb" -o "$STRUCTURES_DIR/${pdb_lower}.pdb" 2>/dev/null; then
                echo -n "."
            else
                echo -e "\n  ${RED}Failed to download $pdb_id${NC}"
            fi
        fi
    done
    echo ""

    DOWNLOADED=$(ls "$STRUCTURES_DIR"/*.{cif,pdb} 2>/dev/null | wc -l)
    echo -e "${GREEN}Downloaded $DOWNLOADED structures${NC}"
fi

# Check if structures exist
N_STRUCTURES=$(ls "$STRUCTURES_DIR"/*.{cif,pdb} 2>/dev/null | wc -l)
if [[ $N_STRUCTURES -eq 0 ]]; then
    echo -e "${RED}No structures found in $STRUCTURES_DIR${NC}"
    echo "Run with --download to fetch structures from RCSB"
    exit 1
fi
echo "Structures available: $N_STRUCTURES"

# Check prism binary
if [[ ! -f "$PRISM_BIN" ]]; then
    echo -e "${RED}Error: prism-lbs binary not found${NC}"
    echo "Run: cargo build --release --features cuda -p prism-lbs"
    exit 1
fi

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
