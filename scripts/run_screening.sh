#!/bin/bash
# ============================================================================
# PRISM-LBS OFFICIAL SCRIPT — DO NOT MODIFY
# ============================================================================
# run_screening.sh — Fast GPU Screening Mode
#
# Ultra-fast initial screening using pure GPU kernel.
# Use this for high-throughput virtual screening of large libraries.
#
# Usage:
#   ./scripts/run_screening.sh <input_dir> <output_dir> [--parallel N]
#
# Options:
#   --parallel N    Number of parallel workers (default: 4)
#
# Example:
#   ./scripts/run_screening.sh benchmarks/datasets/cryptobench/structures/test/ /tmp/screening/
#
# LOCAL DATA ONLY — No auto-downloads. No network access. Ever.
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Environment
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.6}"
export PRISM_PTX_DIR="${PRISM_PTX_DIR:-$PROJECT_ROOT/target/ptx}"
export RUST_LOG="${RUST_LOG:-warn}"

PRISM_BIN="$PROJECT_ROOT/target/release/prism-lbs"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# Defaults
PARALLEL=4
INPUT_DIR=""
OUTPUT_DIR=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel|-p)
            PARALLEL="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 <input_dir> <output_dir> [--parallel N]"
            echo ""
            echo "Fast GPU screening for high-throughput virtual screening."
            exit 0
            ;;
        -*)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
        *)
            if [[ -z "$INPUT_DIR" ]]; then
                INPUT_DIR="$1"
            elif [[ -z "$OUTPUT_DIR" ]]; then
                OUTPUT_DIR="$1"
            else
                echo -e "${RED}Too many arguments${NC}"
                exit 1
            fi
            shift
            ;;
    esac
done

# Validate
if [[ -z "$INPUT_DIR" ]] || [[ -z "$OUTPUT_DIR" ]]; then
    echo -e "${RED}ERROR: Both input and output directories required${NC}"
    echo "Usage: $0 <input_dir> <output_dir> [--parallel N]"
    exit 1
fi

if [[ ! -d "$INPUT_DIR" ]]; then
    echo -e "${RED}ERROR: Input directory does not exist: $INPUT_DIR${NC}"
    exit 1
fi

if [[ ! -f "$PRISM_BIN" ]]; then
    echo -e "${RED}ERROR: prism-lbs binary not found at $PRISM_BIN${NC}"
    echo "Please build with: cargo build --release --features cuda -p prism-lbs"
    exit 1
fi

N_STRUCTURES=$(find "$INPUT_DIR" -maxdepth 1 \( -name "*.pdb" -o -name "*.cif" \) -type f 2>/dev/null | wc -l)
if [[ "$N_STRUCTURES" -eq 0 ]]; then
    echo -e "${RED}ERROR: No PDB/CIF files found in $INPUT_DIR${NC}"
    exit 1
fi

echo ""
echo "============================================================"
echo "   PRISM-LBS Fast GPU Screening"
echo "============================================================"
echo ""
echo "Input:      $INPUT_DIR"
echo "Output:     $OUTPUT_DIR"
echo "Structures: $N_STRUCTURES"
echo "Parallel:   $PARALLEL workers"
echo "Mode:       --pure-gpu (mega-fused GPU kernel)"
echo ""

mkdir -p "$OUTPUT_DIR"
START_TIME=$(date +%s)

"$PRISM_BIN" \
    -i "$INPUT_DIR" \
    -o "$OUTPUT_DIR" \
    --format json \
    --pure-gpu \
    batch --parallel "$PARALLEL"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
N_RESULTS=$(find "$OUTPUT_DIR" -maxdepth 1 -name "*.json" -type f 2>/dev/null | wc -l)

echo ""
echo -e "${GREEN}Screening complete: $N_RESULTS structures in ${DURATION}s${NC}"
if [[ $DURATION -gt 0 ]]; then
    RATE=$(echo "scale=2; $N_RESULTS / $DURATION" | bc)
    echo -e "${GREEN}Throughput: ${RATE} structures/sec${NC}"
fi
echo ""
echo "Results: $OUTPUT_DIR"
echo ""
echo "Next step: Run refinement on top hits:"
echo "  ./scripts/refine_top_hits.sh $OUTPUT_DIR --top 50"
echo ""
