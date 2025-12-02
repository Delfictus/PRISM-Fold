#!/bin/bash
#
# PRISM-LBS LIGYSIS Benchmark Runner
# Runs full 3,446+ structure benchmark against LIGYSIS ground truth
#
# External Ground Truth Source:
#   - https://zenodo.org/doi/10.5281/zenodo.13171100
#   - Journal of Cheminformatics 2024
#
# Usage:
#   ./scripts/run_ligysis_benchmark.sh [--quick] [--skip-run] [--download]
#
# Options:
#   --quick      Run on subset (100 structures) for quick testing
#   --skip-run   Skip PRISM prediction, only evaluate existing predictions
#   --download   Force download of structures from RCSB
#
# PRISM-LBS Mode:
#   Uses --unified for combined detector (geometric + softspot cryptic sites)
#   Uses --gpu-geometry for GPU-accelerated surface calculations
#   Uses --publication for publication-quality output format
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Paths
LIGYSIS_DIR="$PROJECT_ROOT/benchmarks/datasets/ligysis"
# Use ORIGINAL structures from RCSB (not transformed) for ground truth comparison
PDB_DIR="$PROJECT_ROOT/benchmarks/datasets/ligysis/structures"
GROUND_TRUTH_PKL="$LIGYSIS_DIR/MASTER_POCKET_SHAPE_DICT_EXTENDED_TRANS.pkl"
RESULTS_DIR="$PROJECT_ROOT/benchmarks/datasets/ligysis/results"
PRISM_BIN="$PROJECT_ROOT/target/release/prism-lbs"
DOWNLOAD_SCRIPT="$SCRIPT_DIR/download_ligysis_structures.py"

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
echo "   (Source: https://zenodo.org/doi/10.5281/zenodo.13171100)"
echo "============================================================"
echo ""

# Parse arguments
QUICK_MODE=false
SKIP_RUN=false
DOWNLOAD_MODE=false
for arg in "$@"; do
    case $arg in
        --quick) QUICK_MODE=true ;;
        --skip-run) SKIP_RUN=true ;;
        --download) DOWNLOAD_MODE=true ;;
    esac
done

if $QUICK_MODE; then
    echo -e "${YELLOW}Running in QUICK mode (100 structures)${NC}"
fi

# Download structures if requested or if directory doesn't exist
if $DOWNLOAD_MODE || [[ ! -d "$PDB_DIR" ]]; then
    echo -e "${BLUE}Downloading original structures from RCSB...${NC}"
    echo "(This is required because ground truth uses original RCSB coordinates)"
    echo ""

    if $QUICK_MODE; then
        python3 "$DOWNLOAD_SCRIPT" --quick --output "$PDB_DIR"
    else
        python3 "$DOWNLOAD_SCRIPT" --output "$PDB_DIR"
    fi
    echo ""
fi

# Check ground truth file
if [[ ! -f "$GROUND_TRUTH_PKL" ]]; then
    echo -e "${RED}Error: Ground truth file not found:${NC}"
    echo "  $GROUND_TRUTH_PKL"
    echo ""
    echo "Please ensure MASTER_POCKET_SHAPE_DICT_EXTENDED_TRANS.pkl is in:"
    echo "  $LIGYSIS_DIR"
    echo ""
    echo "Original source: https://zenodo.org/doi/10.5281/zenodo.13171100"
    exit 1
fi

echo -e "${GREEN}Ground truth:${NC} $(basename $GROUND_TRUTH_PKL)"
echo "  Size: $(du -h "$GROUND_TRUTH_PKL" | cut -f1)"

# Check prerequisites
if [[ ! -f "$PRISM_BIN" ]]; then
    echo -e "${RED}Error: prism-lbs binary not found at $PRISM_BIN${NC}"
    echo "Run: cargo build --release --features cuda -p prism-lbs"
    exit 1
fi

if [[ ! -d "$PDB_DIR" ]]; then
    echo -e "${RED}Error: LIGYSIS structures directory not found at $PDB_DIR${NC}"
    echo "Run with --download to fetch structures from RCSB"
    exit 1
fi

# Count structures (both CIF and PDB files)
TOTAL_STRUCTURES=$(ls -1 "$PDB_DIR"/*.{cif,pdb} 2>/dev/null | wc -l)
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
**Duration:** ${DURATION}s ($(echo "scale=2; $PROCESSED / $DURATION" | bc) structures/sec)

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

## Full Results Matrix

\`\`\`
$(cat "$RUN_DIR/evaluation_log.txt" 2>/dev/null | grep -A20 "Full Results Matrix" || echo "See evaluation.csv")
\`\`\`

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
