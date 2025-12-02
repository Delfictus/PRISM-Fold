#!/bin/bash
# ============================================================================
# PRISM-LBS OFFICIAL SCRIPT — DO NOT MODIFY
# ============================================================================
# run_ultra_refinement.sh — UltraPrecise Refinement Mode
#
# Maximum precision analysis using unified detector with GPU geometry.
# Use this for final analysis of promising hits.
#
# Usage:
#   ./scripts/run_ultra_refinement.sh <input_dir> <output_dir> [--parallel N]
#
# Options:
#   --parallel N    Number of parallel workers (default: 4)
#
# Example:
#   ./scripts/run_ultra_refinement.sh /tmp/top_hits/ /tmp/refined/
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
            echo "UltraPrecise refinement for final hit analysis."
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
echo "   PRISM-LBS UltraPrecise Refinement"
echo "============================================================"
echo ""
echo "Input:      $INPUT_DIR"
echo "Output:     $OUTPUT_DIR"
echo "Structures: $N_STRUCTURES"
echo "Parallel:   $PARALLEL workers"
echo "Mode:       --unified --gpu-geometry --precision high_precision --publication"
echo ""

mkdir -p "$OUTPUT_DIR"
START_TIME=$(date +%s)

"$PRISM_BIN" \
    -i "$INPUT_DIR" \
    -o "$OUTPUT_DIR" \
    --format json \
    --unified \
    --gpu-geometry \
    --precision high_precision \
    --publication \
    batch --parallel "$PARALLEL"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
N_RESULTS=$(find "$OUTPUT_DIR" -maxdepth 1 -name "*.json" -type f 2>/dev/null | wc -l)

# Generate .METADATA files
for json_file in "$OUTPUT_DIR"/*.json; do
    if [[ -f "$json_file" ]]; then
        metadata_file="${json_file}.METADATA"
        cat > "$metadata_file" << EOF
{
  "source": "PRISM-LBS UltraPrecise Refinement",
  "version": "$(git -C "$PROJECT_ROOT" describe --tags --always 2>/dev/null || echo 'unknown')",
  "timestamp": "$(date -Iseconds)",
  "mode": "ultra_precise",
  "flags": "--unified --gpu-geometry --precision high_precision --publication",
  "cuda_home": "$CUDA_HOME",
  "ptx_dir": "$PRISM_PTX_DIR"
}
EOF
    fi
done

echo ""
echo -e "${GREEN}Refinement complete: $N_RESULTS structures in ${DURATION}s${NC}"
echo ""
echo "Results: $OUTPUT_DIR"
echo "Metadata: $OUTPUT_DIR/*.json.METADATA"
echo ""
