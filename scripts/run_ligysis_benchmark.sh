#!/bin/bash
# ============================================================================
# PRISM-LBS BENCHMARK SCRIPT — NO AUTO-DOWNLOADS
# This script will NOT download data. You must provide it manually.
# ============================================================================
#
# LIGYSIS External Ground Truth Benchmark Runner
#
# REQUIRED DATA (must be manually provided):
#   - benchmarks/datasets/ligysis/MASTER_POCKET_SHAPE_DICT_EXTENDED_TRANS.pkl
#   - benchmarks/datasets/ligysis/structures/*.cif or *.pdb
#
# Download sources:
#   - Ground truth: https://zenodo.org/doi/10.5281/zenodo.13171100
#   - Structures: https://files.rcsb.org/download/{pdb_id}.cif
#
# Usage:
#   ./scripts/run_ligysis_benchmark.sh [--quick] [--skip-run]
#
# Options:
#   --quick      Run on subset (100 structures) for quick testing
#   --skip-run   Skip PRISM prediction, only evaluate existing predictions
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Paths
LIGYSIS_DIR="$PROJECT_ROOT/benchmarks/datasets/ligysis"
PDB_DIR="$PROJECT_ROOT/benchmarks/datasets/ligysis/structures"
GROUND_TRUTH_PKL="$LIGYSIS_DIR/MASTER_POCKET_SHAPE_DICT_EXTENDED_TRANS.pkl"
RESULTS_DIR="$PROJECT_ROOT/benchmarks/datasets/ligysis/results"
PRISM_BIN="$PROJECT_ROOT/target/release/prism-lbs"

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
echo "   LIGYSIS External Ground Truth Benchmark"
echo "   (NO AUTO-DOWNLOADS - Manual data provision required)"
echo "============================================================"
echo ""

# Parse arguments
QUICK_MODE=false
SKIP_RUN=false
for arg in "$@"; do
    case $arg in
        --quick) QUICK_MODE=true ;;
        --skip-run) SKIP_RUN=true ;;
        --download)
            echo -e "${RED}ERROR: --download flag is not supported.${NC}"
            echo "This script does NOT auto-download data."
            echo "Please manually download structures to: $PDB_DIR/"
            echo "Source: https://files.rcsb.org/download/{pdb_id}.cif"
            exit 1
            ;;
    esac
done

if $QUICK_MODE; then
    echo -e "${YELLOW}Running in QUICK mode (100 structures)${NC}"
fi

# ============================================================================
# MANDATORY DATA CHECKS - Exit with error if missing
# ============================================================================

# Check ground truth file
if [[ ! -f "$GROUND_TRUTH_PKL" ]]; then
    echo -e "${RED}ERROR: Missing required benchmark dataset: LIGYSIS ground truth${NC}"
    echo ""
    echo "Please place it in: $GROUND_TRUTH_PKL"
    echo ""
    echo "Download from: https://zenodo.org/doi/10.5281/zenodo.13171100"
    exit 1
fi

# Check structures directory
if [[ ! -d "$PDB_DIR" ]]; then
    echo -e "${RED}ERROR: Missing required structures directory${NC}"
    echo ""
    echo "Please create and populate: $PDB_DIR/"
    echo ""
    echo "Download structures from RCSB:"
    echo "  https://files.rcsb.org/download/{pdb_id}.cif"
    exit 1
fi

# Check if structures exist
TOTAL_STRUCTURES=$(ls -1 "$PDB_DIR"/*.{cif,pdb} 2>/dev/null | wc -l || echo "0")
if [[ "$TOTAL_STRUCTURES" -eq 0 ]]; then
    echo -e "${RED}ERROR: No structure files found in $PDB_DIR${NC}"
    echo ""
    echo "Please download CIF/PDB files to this directory."
    echo "Source: https://files.rcsb.org/download/{pdb_id}.cif"
    exit 1
fi

# Check prism binary
if [[ ! -f "$PRISM_BIN" ]]; then
    echo -e "${RED}ERROR: prism-lbs binary not found at $PRISM_BIN${NC}"
    echo ""
    echo "Please build with: cargo build --release --features cuda -p prism-lbs"
    exit 1
fi

echo -e "${GREEN}All required data present${NC}"
echo ""
echo -e "${GREEN}Ground truth:${NC} $(basename $GROUND_TRUTH_PKL)"
echo "  Size: $(du -h "$GROUND_TRUTH_PKL" | cut -f1)"
echo "Structures available: $TOTAL_STRUCTURES"

# Setup
mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$RESULTS_DIR/run_$TIMESTAMP"
JSON_OUTPUT="$RUN_DIR/predictions"
EVAL_OUTPUT="$RUN_DIR/evaluation.csv"
REPORT_OUTPUT="$RUN_DIR/BENCHMARK_REPORT.md"

mkdir -p "$JSON_OUTPUT"

echo ""
echo "Run directory: $RUN_DIR"
echo ""

# Run PRISM predictions
if $QUICK_MODE; then
    # Copy first 100 structures to temp dir
    TEMP_INPUT=$(mktemp -d)
    echo "Copying 100 structures to $TEMP_INPUT..."
    ls -1 "$PDB_DIR"/*.{cif,pdb} 2>/dev/null | head -100 | xargs -I{} cp {} "$TEMP_INPUT/"
    INPUT_DIR="$TEMP_INPUT"
    EXPECTED=$(ls -1 "$TEMP_INPUT"/*.{cif,pdb} 2>/dev/null | wc -l)
else
    INPUT_DIR="$PDB_DIR"
    EXPECTED=$TOTAL_STRUCTURES
fi

if $SKIP_RUN; then
    echo -e "${YELLOW}Skipping PRISM run, using existing predictions...${NC}"
    # Find most recent predictions directory
    LATEST_RUN=$(ls -td "$RESULTS_DIR"/run_* 2>/dev/null | head -1)
    if [[ -z "$LATEST_RUN" ]] || [[ ! -d "$LATEST_RUN/predictions" ]]; then
        echo -e "${RED}No existing predictions found in $RESULTS_DIR${NC}"
        exit 1
    fi
    JSON_OUTPUT="$LATEST_RUN/predictions"
    RUN_DIR="$LATEST_RUN"
    PROCESSED=$(ls -1 "$JSON_OUTPUT"/*.json 2>/dev/null | wc -l)
    DURATION=0
    echo "Using: $JSON_OUTPUT"
    echo "Found $PROCESSED predictions"
else
    echo -e "${GREEN}Starting PRISM-LBS predictions on $EXPECTED structures...${NC}"
    echo ""

    START_TIME=$(date +%s)

    # Run with GPU, unified detector (geometric + softspot), and publication format
    "$PRISM_BIN" \
        -i "$INPUT_DIR" \
        -o "$JSON_OUTPUT" \
        --format json \
        --publication \
        --gpu-geometry \
        --unified \
        batch --parallel 4

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    PROCESSED=$(ls -1 "$JSON_OUTPUT"/*.json 2>/dev/null | wc -l)
    echo ""
    echo -e "${GREEN}Completed: $PROCESSED/$EXPECTED structures in ${DURATION}s${NC}"
    if [[ $DURATION -gt 0 ]]; then
        echo "  Throughput: $(echo "scale=2; $PROCESSED / $DURATION" | bc) structures/sec"
    fi
fi

# Run evaluation using external ground truth
echo ""
echo -e "${GREEN}Evaluating against LIGYSIS external ground truth...${NC}"
echo ""

python3 "$SCRIPT_DIR/ligysis_evaluate.py" \
    --ground-truth "$GROUND_TRUTH_PKL" \
    --predictions "$JSON_OUTPUT" \
    --output "$RUN_DIR/evaluation_results.json" \
    --csv "$EVAL_OUTPUT" \
    2>&1 | tee "$RUN_DIR/evaluation_log.txt"

# Generate markdown report
cat > "$REPORT_OUTPUT" << EOF
# PRISM-LBS LIGYSIS Benchmark Report

**Date:** $(date -Iseconds)
**Structures:** $PROCESSED / $EXPECTED
**Duration:** ${DURATION}s

## Configuration

- GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "N/A")
- Mode: Unified (Geometric + Softspot Cryptic Site Detection)
- GPU Geometry: Enabled
- PTX Directory: $PRISM_PTX_DIR
- CUDA Home: $CUDA_HOME

## Results

### Top-N+2 Recall @ IRel=0.25 (LIGYSIS Primary Metric)

| Method | Recall (%) |
|--------|-----------|
$(awk -F',' 'NR>1 && $2==2 && $4==0.25 {printf "| %s | %.1f |\n", $1, $6}' "$EVAL_OUTPUT" 2>/dev/null || echo "| PRISM-LBS | N/A |")

### Competitor Baselines (from LIGYSIS paper)

| Method | Recall (%) |
|--------|-----------|
| fpocket | 60 |
| P2Rank | 54 |
| DeepPocket | 60 |
| VN-EGNN | 46 |
| PUResNet | 45 |
| IF-SitePred | 55 |

## Files

- Predictions: \`$JSON_OUTPUT/\`
- Evaluation CSV: \`$EVAL_OUTPUT\`
- Log: \`$RUN_DIR/evaluation_log.txt\`

---
Generated by PRISM-LBS LIGYSIS Benchmark Runner
EOF

echo ""
echo "============================================================"
echo "   Benchmark Complete"
echo "============================================================"
echo ""
echo "Results saved to: $RUN_DIR"
echo ""
echo "Key files:"
echo "  - Report: $REPORT_OUTPUT"
echo "  - Evaluation: $EVAL_OUTPUT"
echo ""

# Cleanup temp dir if quick mode
if $QUICK_MODE && [[ -n "$TEMP_INPUT" ]]; then
    rm -rf "$TEMP_INPUT"
fi

# Print summary
echo ""
if [[ -f "$EVAL_OUTPUT" ]]; then
    TOP_N2=$(awk -F',' 'NR>1 && $2==2 && $4==0.25 {print $6}' "$EVAL_OUTPUT" 2>/dev/null)
    if [[ -n "$TOP_N2" ]]; then
        if (( $(echo "$TOP_N2 >= 60" | bc -l) )); then
            echo -e "${GREEN}SUCCESS: PRISM-LBS achieves ${TOP_N2}% recall (>= 60% target)${NC}"
        else
            echo -e "${YELLOW}PENDING: PRISM-LBS at ${TOP_N2}% recall (target: 60%)${NC}"
        fi
    fi
fi
