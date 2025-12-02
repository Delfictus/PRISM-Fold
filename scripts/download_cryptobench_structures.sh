#!/bin/bash
#
# Download CryptoBench test structures from RCSB
# Run this once, then structures are cached locally
#
# Usage: ./scripts/download_cryptobench_structures.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_ROOT/benchmarks/datasets/cryptobench/structures/test"
PDB_LIST="$PROJECT_ROOT/benchmarks/datasets/cryptobench/pdb_ids_test.txt"

echo "============================================================"
echo "   CryptoBench Structure Downloader"
echo "============================================================"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR"

# Check if PDB list exists
if [[ ! -f "$PDB_LIST" ]]; then
    echo "Generating PDB ID list from folds.json..."
    python3 -c "
import json
with open('$PROJECT_ROOT/benchmarks/datasets/cryptobench/folds.json') as f:
    folds = json.load(f)
for pdb in folds['test']:
    print(pdb.lower())
" > "$PDB_LIST"
fi

TOTAL=$(wc -l < "$PDB_LIST")
echo "Total structures to download: $TOTAL"
echo ""

SUCCESS=0
CACHED=0
FAILED=0

while read pdb; do
    pdb_lower=$(echo "$pdb" | tr '[:upper:]' '[:lower:]')

    # Check if already exists
    if [[ -f "$OUTPUT_DIR/${pdb_lower}.cif" ]] || [[ -f "$OUTPUT_DIR/${pdb_lower}.pdb" ]]; then
        ((CACHED++))
        continue
    fi

    # Try CIF first (smaller), then PDB
    if curl -sf "https://files.rcsb.org/download/${pdb_lower}.cif" -o "$OUTPUT_DIR/${pdb_lower}.cif" 2>/dev/null; then
        ((SUCCESS++))
        echo -n "."
    elif curl -sf "https://files.rcsb.org/download/${pdb_lower}.pdb" -o "$OUTPUT_DIR/${pdb_lower}.pdb" 2>/dev/null; then
        ((SUCCESS++))
        echo -n "."
    else
        ((FAILED++))
        echo ""
        echo "  [FAIL] $pdb_lower"
    fi

    # Progress every 50
    if (( (SUCCESS + FAILED) % 50 == 0 )); then
        echo " [$((SUCCESS + CACHED))/$TOTAL]"
    fi
done < "$PDB_LIST"

echo ""
echo ""
echo "============================================================"
echo "   Download Complete"
echo "============================================================"
echo ""
echo "  New downloads: $SUCCESS"
echo "  Already cached: $CACHED"
echo "  Failed: $FAILED"
echo "  Total available: $((SUCCESS + CACHED))"
echo ""
echo "Structures saved to: $OUTPUT_DIR"
echo ""
