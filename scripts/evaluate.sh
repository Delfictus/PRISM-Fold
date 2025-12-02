#!/bin/bash
# ============================================================================
# PRISM-LBS OFFICIAL SCRIPT — DO NOT MODIFY
# ============================================================================
# evaluate.sh — Auto-Detecting Benchmark Evaluator
#
# Automatically detects benchmark type and runs the correct evaluator.
# Supports: CryptoBench, LIGYSIS, or custom ground truth files.
#
# Usage:
#   ./scripts/evaluate.sh <predictions_dir> [--benchmark TYPE] [--output FILE]
#
# Options:
#   --benchmark TYPE    Force benchmark type: cryptobench, ligysis, or custom
#   --output FILE       Output results file (default: evaluation_results.json)
#   --ground-truth FILE Custom ground truth file (for custom benchmarks)
#
# Example:
#   ./scripts/evaluate.sh /tmp/predictions/
#   ./scripts/evaluate.sh /tmp/predictions/ --benchmark cryptobench
#
# LOCAL DATA ONLY — No auto-downloads. No network access. Ever.
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Paths
CRYPTOBENCH_DIR="$PROJECT_ROOT/benchmarks/datasets/cryptobench"
LIGYSIS_DIR="$PROJECT_ROOT/benchmarks/datasets/ligysis"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Defaults
PREDICTIONS_DIR=""
BENCHMARK_TYPE=""
OUTPUT_FILE=""
GROUND_TRUTH=""
SPLIT="test"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --benchmark|-b)
            BENCHMARK_TYPE="$2"
            shift 2
            ;;
        --output|-o)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --ground-truth|-g)
            GROUND_TRUTH="$2"
            shift 2
            ;;
        --split|-s)
            SPLIT="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 <predictions_dir> [--benchmark TYPE] [--output FILE]"
            echo ""
            echo "Auto-detecting benchmark evaluator."
            echo ""
            echo "Options:"
            echo "  --benchmark TYPE    Force benchmark type: cryptobench, ligysis"
            echo "  --output FILE       Output results file"
            echo "  --ground-truth FILE Custom ground truth file"
            echo "  --split SPLIT       Test split to use (default: test)"
            exit 0
            ;;
        -*)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
        *)
            if [[ -z "$PREDICTIONS_DIR" ]]; then
                PREDICTIONS_DIR="$1"
            else
                echo -e "${RED}Too many arguments${NC}"
                exit 1
            fi
            shift
            ;;
    esac
done

# Validate
if [[ -z "$PREDICTIONS_DIR" ]]; then
    echo -e "${RED}ERROR: No predictions directory specified${NC}"
    echo "Usage: $0 <predictions_dir> [--benchmark TYPE]"
    exit 1
fi

if [[ ! -d "$PREDICTIONS_DIR" ]]; then
    echo -e "${RED}ERROR: Predictions directory does not exist: $PREDICTIONS_DIR${NC}"
    exit 1
fi

N_PREDICTIONS=$(find "$PREDICTIONS_DIR" -maxdepth 1 -name "*.json" -type f 2>/dev/null | wc -l)
if [[ "$N_PREDICTIONS" -eq 0 ]]; then
    echo -e "${RED}ERROR: No JSON prediction files found in $PREDICTIONS_DIR${NC}"
    exit 1
fi

echo ""
echo "============================================================"
echo "   PRISM-LBS Benchmark Evaluator"
echo "============================================================"
echo ""
echo "Predictions: $PREDICTIONS_DIR ($N_PREDICTIONS files)"
echo ""

# Auto-detect benchmark type if not specified
if [[ -z "$BENCHMARK_TYPE" ]]; then
    echo -e "${BLUE}Auto-detecting benchmark type...${NC}"

    # Check for CryptoBench
    if [[ -f "$CRYPTOBENCH_DIR/dataset.json" ]] && [[ -f "$CRYPTOBENCH_DIR/folds.json" ]]; then
        # Check if any prediction matches CryptoBench test set
        CRYPTOBENCH_MATCH=$(python3 << PYTHON
import json
from pathlib import Path

predictions = set(p.stem.lower() for p in Path("$PREDICTIONS_DIR").glob("*.json"))
with open("$CRYPTOBENCH_DIR/folds.json") as f:
    folds = json.load(f)
test_pdbs = set(p.lower() for p in folds.get("test", []))
overlap = len(predictions & test_pdbs)
print(overlap)
PYTHON
        )
        if [[ "$CRYPTOBENCH_MATCH" -gt 10 ]]; then
            BENCHMARK_TYPE="cryptobench"
            echo -e "${GREEN}Detected: CryptoBench (${CRYPTOBENCH_MATCH} matching structures)${NC}"
        fi
    fi

    # Check for LIGYSIS
    if [[ -z "$BENCHMARK_TYPE" ]] && [[ -f "$LIGYSIS_DIR/MASTER_POCKET_SHAPE_DICT_EXTENDED_TRANS.pkl" ]]; then
        BENCHMARK_TYPE="ligysis"
        echo -e "${GREEN}Detected: LIGYSIS${NC}"
    fi

    if [[ -z "$BENCHMARK_TYPE" ]]; then
        echo -e "${RED}ERROR: Could not auto-detect benchmark type${NC}"
        echo "Please specify with --benchmark cryptobench or --benchmark ligysis"
        echo ""
        echo "Required files:"
        echo "  CryptoBench: $CRYPTOBENCH_DIR/dataset.json, folds.json"
        echo "  LIGYSIS:     $LIGYSIS_DIR/MASTER_POCKET_SHAPE_DICT_EXTENDED_TRANS.pkl"
        exit 1
    fi
fi

echo "Benchmark:   $BENCHMARK_TYPE"
echo "Split:       $SPLIT"
echo ""

# Set default output file
if [[ -z "$OUTPUT_FILE" ]]; then
    OUTPUT_FILE="${PREDICTIONS_DIR%/}_${BENCHMARK_TYPE}_results.json"
fi

# Run appropriate evaluator
case "$BENCHMARK_TYPE" in
    cryptobench)
        # Check required files
        if [[ ! -f "$CRYPTOBENCH_DIR/dataset.json" ]]; then
            echo -e "${RED}ERROR: Missing CryptoBench dataset.json${NC}"
            echo "Please place it in: $CRYPTOBENCH_DIR/dataset.json"
            echo "Download from: https://osf.io/pz4a9/"
            exit 1
        fi
        if [[ ! -f "$CRYPTOBENCH_DIR/folds.json" ]]; then
            echo -e "${RED}ERROR: Missing CryptoBench folds.json${NC}"
            echo "Please place it in: $CRYPTOBENCH_DIR/folds.json"
            echo "Download from: https://osf.io/pz4a9/"
            exit 1
        fi

        echo -e "${GREEN}Running CryptoBench evaluation...${NC}"
        echo ""

        python3 "$SCRIPT_DIR/cryptobench_evaluate.py" \
            --predictions "$PREDICTIONS_DIR" \
            --split "$SPLIT" \
            --output "$OUTPUT_FILE"
        ;;

    ligysis)
        LIGYSIS_GT="$LIGYSIS_DIR/MASTER_POCKET_SHAPE_DICT_EXTENDED_TRANS.pkl"
        if [[ ! -f "$LIGYSIS_GT" ]]; then
            echo -e "${RED}ERROR: Missing LIGYSIS ground truth${NC}"
            echo "Please place it in: $LIGYSIS_GT"
            echo "Download from: https://zenodo.org/doi/10.5281/zenodo.13171100"
            exit 1
        fi

        echo -e "${GREEN}Running LIGYSIS evaluation...${NC}"
        echo ""

        python3 "$SCRIPT_DIR/ligysis_evaluate.py" \
            --ground-truth "$LIGYSIS_GT" \
            --predictions "$PREDICTIONS_DIR" \
            --output "$OUTPUT_FILE"
        ;;

    custom)
        if [[ -z "$GROUND_TRUTH" ]]; then
            echo -e "${RED}ERROR: Custom benchmark requires --ground-truth FILE${NC}"
            exit 1
        fi
        if [[ ! -f "$GROUND_TRUTH" ]]; then
            echo -e "${RED}ERROR: Ground truth file not found: $GROUND_TRUTH${NC}"
            exit 1
        fi

        echo -e "${YELLOW}Custom benchmark evaluation not yet implemented${NC}"
        echo "Ground truth: $GROUND_TRUTH"
        exit 1
        ;;

    *)
        echo -e "${RED}ERROR: Unknown benchmark type: $BENCHMARK_TYPE${NC}"
        echo "Supported: cryptobench, ligysis, custom"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "   Evaluation Complete"
echo "============================================================"
echo ""
echo "Results saved to: $OUTPUT_FILE"
echo ""

# Print summary if file exists
if [[ -f "$OUTPUT_FILE" ]]; then
    echo -e "${BLUE}Summary:${NC}"
    python3 << PYTHON
import json
from pathlib import Path

try:
    with open("$OUTPUT_FILE") as f:
        results = json.load(f)

    # Print key metrics
    if 'metrics' in results:
        m = results['metrics']
        if 'dcc_success_rate' in m:
            print(f"  DCC Success Rate: {m['dcc_success_rate']*100:.1f}%")
        if 'recall_at_n2' in m:
            print(f"  Recall @ N+2: {m['recall_at_n2']*100:.1f}%")
        if 'top1_accuracy' in m:
            print(f"  Top-1 Accuracy: {m['top1_accuracy']*100:.1f}%")
    elif 'summary' in results:
        s = results['summary']
        for k, v in s.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
except Exception as e:
    print(f"  Could not parse results: {e}")
PYTHON
    echo ""
fi
