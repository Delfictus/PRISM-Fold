#!/bin/bash
# ============================================================================
# PRISM-LBS OFFICIAL SCRIPT — DO NOT MODIFY
# ============================================================================
# refine_top_hits.sh — Universal Top-Hit Refinement Workflow
#
# Takes screening results and re-runs top N structures in UltraPrecise mode.
# Works on any benchmark (CryptoBench, LIGYSIS, custom folders).
#
# Usage:
#   ./scripts/refine_top_hits.sh <screening_results_dir> [--top N] [--output DIR]
#
# Options:
#   --top N      Number of top structures to refine (default: 50)
#   --output     Output directory (default: <input>_refined/)
#
# Example:
#   ./scripts/refine_top_hits.sh /tmp/screening_results/ --top 100
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
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Defaults
TOP_N=50
INPUT_DIR=""
OUTPUT_DIR=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --top)
            TOP_N="$2"
            shift 2
            ;;
        --output|-o)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 <screening_results_dir> [--top N] [--output DIR]"
            echo ""
            echo "Refine top N structures from screening results in UltraPrecise mode."
            echo ""
            echo "Options:"
            echo "  --top N      Number of top structures to refine (default: 50)"
            echo "  --output     Output directory (default: <input>_refined/)"
            exit 0
            ;;
        -*)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
        *)
            if [[ -z "$INPUT_DIR" ]]; then
                INPUT_DIR="$1"
            else
                echo -e "${RED}Too many arguments${NC}"
                exit 1
            fi
            shift
            ;;
    esac
done

# Validate input
if [[ -z "$INPUT_DIR" ]]; then
    echo -e "${RED}ERROR: No input directory specified${NC}"
    echo "Usage: $0 <screening_results_dir> [--top N] [--output DIR]"
    exit 1
fi

if [[ ! -d "$INPUT_DIR" ]]; then
    echo -e "${RED}ERROR: Input directory does not exist: $INPUT_DIR${NC}"
    exit 1
fi

# Check for JSON files
JSON_COUNT=$(find "$INPUT_DIR" -maxdepth 1 -name "*.json" -type f 2>/dev/null | wc -l)
if [[ "$JSON_COUNT" -eq 0 ]]; then
    echo -e "${RED}ERROR: No JSON prediction files found in $INPUT_DIR${NC}"
    exit 1
fi

# Check prism binary
if [[ ! -f "$PRISM_BIN" ]]; then
    echo -e "${RED}ERROR: prism-lbs binary not found at $PRISM_BIN${NC}"
    echo "Please build with: cargo build --release --features cuda -p prism-lbs"
    exit 1
fi

# Set default output directory
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="${INPUT_DIR%/}_refined"
fi

echo ""
echo "============================================================"
echo "   PRISM-LBS Top-Hit Refinement Pipeline"
echo "============================================================"
echo ""
echo -e "${BLUE}Input:${NC}  $INPUT_DIR"
echo -e "${BLUE}Output:${NC} $OUTPUT_DIR"
echo -e "${BLUE}Top N:${NC}  $TOP_N"
echo ""

# Step 1: Rank all predictions by druggability_score.total
echo -e "${GREEN}Step 1: Ranking predictions by druggability score...${NC}"

RANKING_FILE=$(mktemp)
STRUCTURES_FILE=$(mktemp)

python3 << PYTHON
import json
import os
import sys
from pathlib import Path

input_dir = Path("$INPUT_DIR")
top_n = $TOP_N

# Collect all predictions with druggability scores
scored = []
for json_file in input_dir.glob("*.json"):
    try:
        with open(json_file) as f:
            data = json.load(f)

        # Handle both single-pocket and multi-pocket formats
        pockets = data.get('pockets', [])
        if not pockets and 'druggability_score' in data:
            # Single pocket format
            pockets = [data]

        # Get best pocket score for this structure
        best_score = 0.0
        best_pocket = None
        for i, pocket in enumerate(pockets):
            score_data = pocket.get('druggability_score', {})
            if isinstance(score_data, dict):
                total = score_data.get('total', 0.0)
            else:
                total = float(score_data) if score_data else 0.0

            if total > best_score:
                best_score = total
                best_pocket = i + 1

        # Get structure file path
        structure_file = data.get('metadata', {}).get('input_file', '')
        if not structure_file:
            # Try to infer from filename
            stem = json_file.stem
            for ext in ['.pdb', '.cif', '.PDB', '.CIF']:
                candidate = input_dir.parent / 'structures' / (stem + ext)
                if candidate.exists():
                    structure_file = str(candidate)
                    break

        if structure_file and os.path.exists(structure_file):
            scored.append({
                'structure': structure_file,
                'json': str(json_file),
                'score': best_score,
                'pocket': best_pocket,
                'pdb_id': json_file.stem
            })
    except Exception as e:
        print(f"  Warning: Could not parse {json_file.name}: {e}", file=sys.stderr)

# Sort by score descending
scored.sort(key=lambda x: x['score'], reverse=True)

# Take top N
top = scored[:top_n]

print(f"Found {len(scored)} predictions, selecting top {len(top)}")
print("")

if top:
    print("Top 5 by druggability score:")
    for i, entry in enumerate(top[:5], 1):
        print(f"  {i}. {entry['pdb_id']}: {entry['score']:.4f} (pocket {entry['pocket']})")
    print("")

# Write structure paths to temp file
with open("$STRUCTURES_FILE", 'w') as f:
    for entry in top:
        f.write(entry['structure'] + '\n')

# Write full ranking
with open("$RANKING_FILE", 'w') as f:
    json.dump(top, f, indent=2)

print(f"Structures to refine: {len(top)}")
PYTHON

N_TO_REFINE=$(wc -l < "$STRUCTURES_FILE")

if [[ "$N_TO_REFINE" -eq 0 ]]; then
    echo -e "${RED}ERROR: No structures found to refine${NC}"
    echo "Ensure screening results contain valid structure file paths."
    rm -f "$RANKING_FILE" "$STRUCTURES_FILE"
    exit 1
fi

echo ""
echo -e "${GREEN}Step 2: Creating refinement input directory...${NC}"

# Create temp directory with symlinks to top structures
REFINE_INPUT=$(mktemp -d)
while IFS= read -r structure; do
    if [[ -f "$structure" ]]; then
        ln -sf "$structure" "$REFINE_INPUT/"
    fi
done < "$STRUCTURES_FILE"

LINKED_COUNT=$(ls -1 "$REFINE_INPUT" 2>/dev/null | wc -l)
echo "Linked $LINKED_COUNT structures for refinement"

echo ""
echo -e "${GREEN}Step 3: Running UltraPrecise refinement...${NC}"
echo "  Mode: --unified --gpu-geometry --precision high_precision --publication"
echo ""

mkdir -p "$OUTPUT_DIR"
START_TIME=$(date +%s)

"$PRISM_BIN" \
    -i "$REFINE_INPUT" \
    -o "$OUTPUT_DIR" \
    --format json \
    --unified \
    --gpu-geometry \
    --precision high_precision \
    --publication \
    batch --parallel 4

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Count results
N_RESULTS=$(find "$OUTPUT_DIR" -maxdepth 1 -name "*.json" -type f 2>/dev/null | wc -l)

echo ""
echo -e "${GREEN}Step 4: Generating metadata...${NC}"

# Create .METADATA companion files
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
  "input_screening": "$INPUT_DIR",
  "top_n_refined": $TOP_N,
  "cuda_home": "$CUDA_HOME",
  "ptx_dir": "$PRISM_PTX_DIR"
}
EOF
    fi
done

# Find best result
BEST_RESULT=$(python3 << PYTHON
import json
from pathlib import Path

output_dir = Path("$OUTPUT_DIR")
best_score = 0.0
best_pdb = ""
best_pocket = 0

for json_file in output_dir.glob("*.json"):
    try:
        with open(json_file) as f:
            data = json.load(f)

        pockets = data.get('pockets', [])
        if not pockets and 'druggability_score' in data:
            pockets = [data]

        for i, pocket in enumerate(pockets):
            score_data = pocket.get('druggability_score', {})
            if isinstance(score_data, dict):
                total = score_data.get('total', 0.0)
            else:
                total = float(score_data) if score_data else 0.0

            if total > best_score:
                best_score = total
                best_pdb = json_file.stem
                best_pocket = i + 1
    except:
        pass

if best_pdb:
    print(f"{best_pdb} pocket {best_pocket}: {best_score:.4f}")
else:
    print("No results")
PYTHON
)

# Cleanup
rm -rf "$REFINE_INPUT"
rm -f "$RANKING_FILE" "$STRUCTURES_FILE"

echo ""
echo "============================================================"
echo "   Refinement Complete"
echo "============================================================"
echo ""
echo -e "${GREEN}Refined:${NC}    $N_RESULTS / $N_TO_REFINE structures"
echo -e "${GREEN}Duration:${NC}   ${DURATION}s"
echo -e "${GREEN}Best hit:${NC}   $BEST_RESULT"
echo -e "${GREEN}Output:${NC}     $OUTPUT_DIR"
echo ""
echo -e "${BLUE}Files:${NC}"
echo "  - Predictions: $OUTPUT_DIR/*.json"
echo "  - Metadata:    $OUTPUT_DIR/*.json.METADATA"
echo ""
