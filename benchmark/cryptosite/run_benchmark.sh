#!/bin/bash
#===============================================================================
# CryptoSite Benchmark Runner
# Compares PRISM-LBS vs fpocket on cryptic site detection
#===============================================================================

set -euo pipefail

# Export CUDA environment for GPU acceleration
export PATH="/home/diddy/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:/usr/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12.6
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:${LD_LIBRARY_PATH:-}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STRUCTURES_DIR="$SCRIPT_DIR/structures"
RESULTS_DIR="$SCRIPT_DIR/results"
GROUND_TRUTH="$SCRIPT_DIR/ground_truth.csv"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Paths to tools (unified TUI with GPU acceleration)
PRISM_BIN="${PRISM_BIN:-./target/release/prism}"
FPOCKET_BIN="${FPOCKET_BIN:-fpocket}"

# Detection threshold: pocket must contain X% of cryptic residues to count as "detected"
DETECTION_THRESHOLD=${DETECTION_THRESHOLD:-0.5}  # 50% overlap

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║          CryptoSite Benchmark Runner                             ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "Detection threshold: ${DETECTION_THRESHOLD} (${DETECTION_THRESHOLD}00% residue overlap)"
echo ""

#===============================================================================
# Helper functions
#===============================================================================

# Extract residues from PRISM JSON output
# NOTE: PRISM outputs 0-indexed residue array indices, not PDB RESSEQ numbers
# We need to convert by reading the PDB file to get actual residue numbers
extract_prism_residues() {
    local json_file=$1
    local pdb_file=$2  # Optional: PDB file for index-to-RESSEQ mapping
    python3 << PYEOF
import json
import sys
import os

try:
    with open('$json_file', 'r') as f:
        data = json.load(f)

    all_residue_indices = set()
    for pocket in data.get('pockets', []):
        for res in pocket.get('residue_indices', []):
            all_residue_indices.add(int(res))

    # If PDB file provided, map indices to actual residue numbers
    pdb_file = '$pdb_file'
    if pdb_file and os.path.exists(pdb_file):
        # Build residue index -> RESSEQ mapping from PDB
        residue_map = {}
        seen_residues = []
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    try:
                        res_seq = int(line[22:26].strip())
                        chain = line[21]
                        res_key = (chain, res_seq)
                        if res_key not in seen_residues:
                            seen_residues.append(res_key)
                            residue_map[len(seen_residues) - 1] = res_seq
                    except:
                        pass

        # Convert indices to actual RESSEQ numbers
        actual_residues = set()
        for idx in all_residue_indices:
            if idx in residue_map:
                actual_residues.add(residue_map[idx])
            else:
                actual_residues.add(idx)  # Fallback to index
        print(' '.join(map(str, sorted(actual_residues))))
    else:
        # No PDB file, output raw indices
        print(' '.join(map(str, sorted(all_residue_indices))))

except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF
}

# Extract residues from fpocket output
extract_fpocket_residues() {
    local pdb_base=$1
    local fpocket_dir="${pdb_base}_out"

    if [ ! -d "$fpocket_dir" ]; then
        echo ""
        return
    fi

    # Parse fpocket's pocket atoms file (in pockets/ subdirectory)
    python3 << PYEOF
import os
import sys

fpocket_dir = '$fpocket_dir'
residues = set()

# fpocket stores pocket files in pockets/ subdirectory
pockets_dir = os.path.join(fpocket_dir, 'pockets')
search_dirs = [pockets_dir, fpocket_dir]

for search_dir in search_dirs:
    if not os.path.isdir(search_dir):
        continue
    for fname in os.listdir(search_dir):
        if fname.endswith('_atm.pdb'):
            with open(os.path.join(search_dir, fname), 'r') as f:
                for line in f:
                    if line.startswith('ATOM') or line.startswith('HETATM'):
                        try:
                            res_num = int(line[22:26].strip())
                            residues.add(res_num)
                        except:
                            pass

print(' '.join(map(str, sorted(residues))))
PYEOF
}

# Calculate overlap between detected and ground truth residues
calculate_overlap() {
    local detected="$1"
    local ground_truth="$2"
    
    python3 << PYEOF
detected = set(map(int, '$detected'.split())) if '$detected'.strip() else set()
ground_truth = set(map(int, '$ground_truth'.replace(',', ' ').split())) if '$ground_truth'.strip() else set()

if not ground_truth:
    print("0.0")
elif not detected:
    print("0.0")
else:
    overlap = len(detected & ground_truth)
    recall = overlap / len(ground_truth)
    print(f"{recall:.3f}")
PYEOF
}

#===============================================================================
# Run benchmarks
#===============================================================================

PRISM_DETECTED=0
FPOCKET_DETECTED=0
TOTAL=0

# Arrays to store results
declare -a RESULTS

echo -e "${BLUE}Running benchmarks on APO structures...${NC}"
echo ""
printf "%-25s %-10s %-10s %-10s %-10s\n" "Protein" "Difficulty" "PRISM" "fpocket" "Winner"
printf "%-25s %-10s %-10s %-10s %-10s\n" "-------" "----------" "-----" "-------" "------"

# Preprocess CSV with Python (handles quoted fields with commas)
TEMP_CSV=$(mktemp)
python3 -c "
import csv
with open('$GROUND_TRUTH', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if not row or row[0].startswith('#'):
            continue
        # Output pipe-separated for bash
        print('|'.join(row))
" > "$TEMP_CSV"

while IFS='|' read -r apo_pdb holo_pdb protein_name cryptic_residues site_desc difficulty; do
    # Skip empty lines
    [[ -z "$apo_pdb" ]] && continue

    apo_file="$STRUCTURES_DIR/apo/${apo_pdb,,}.pdb"
    
    if [ ! -f "$apo_file" ]; then
        echo -e "${YELLOW}[SKIP]${NC} $apo_pdb not found"
        continue
    fi
    
    ((TOTAL++)) || true

    # Run PRISM (full GPU + unified detector)
    prism_output="$RESULTS_DIR/prism/${apo_pdb,,}.json"
    if [ -f "$PRISM_BIN" ]; then
        PRISM_PTX_DIR="${SCRIPT_DIR}/../../target/ptx" \
        $PRISM_BIN --input "$apo_file" --output "$prism_output" \
            --gpu-geometry --publication --unified 2>/dev/null || true
    fi
    
    # Run fpocket
    fpocket_base="$RESULTS_DIR/fpocket/${apo_pdb,,}"
    mkdir -p "$RESULTS_DIR/fpocket"
    cp "$apo_file" "${fpocket_base}.pdb" 2>/dev/null || true
    (cd "$RESULTS_DIR/fpocket" && $FPOCKET_BIN -f "${apo_pdb,,}.pdb" 2>/dev/null) || true
    
    # Extract detected residues
    prism_residues=""
    if [ -f "$prism_output" ]; then
        prism_residues=$(extract_prism_residues "$prism_output" "$apo_file")
    fi
    
    fpocket_residues=$(extract_fpocket_residues "$fpocket_base")
    
    # Calculate overlap with ground truth
    prism_overlap=$(calculate_overlap "$prism_residues" "$cryptic_residues")
    fpocket_overlap=$(calculate_overlap "$fpocket_residues" "$cryptic_residues")
    
    # Determine detection
    prism_detected="NO"
    fpocket_detected="NO"
    
    if (( $(echo "$prism_overlap >= $DETECTION_THRESHOLD" | bc -l) )); then
        prism_detected="YES"
        ((PRISM_DETECTED++)) || true
    fi
    
    if (( $(echo "$fpocket_overlap >= $DETECTION_THRESHOLD" | bc -l) )); then
        fpocket_detected="YES"
        ((FPOCKET_DETECTED++)) || true
    fi
    
    # Determine winner
    winner="-"
    if [ "$prism_detected" = "YES" ] && [ "$fpocket_detected" = "NO" ]; then
        winner="${GREEN}PRISM${NC}"
    elif [ "$fpocket_detected" = "YES" ] && [ "$prism_detected" = "NO" ]; then
        winner="${RED}fpocket${NC}"
    elif [ "$prism_detected" = "YES" ] && [ "$fpocket_detected" = "YES" ]; then
        if (( $(echo "$prism_overlap > $fpocket_overlap" | bc -l) )); then
            winner="${GREEN}PRISM${NC}"
        elif (( $(echo "$fpocket_overlap > $prism_overlap" | bc -l) )); then
            winner="${RED}fpocket${NC}"
        else
            winner="TIE"
        fi
    fi
    
    # Color the results
    prism_color="${RED}"
    [ "$prism_detected" = "YES" ] && prism_color="${GREEN}"
    
    fpocket_color="${RED}"
    [ "$fpocket_detected" = "YES" ] && fpocket_color="${GREEN}"
    
    # Truncate protein name
    short_name="${protein_name:0:23}"
    
    printf "%-25s %-10s ${prism_color}%-10s${NC} ${fpocket_color}%-10s${NC} %-10b\n" \
        "$short_name" "$difficulty" "${prism_overlap}" "${fpocket_overlap}" "$winner"

done < "$TEMP_CSV"
rm -f "$TEMP_CSV"

#===============================================================================
# Summary
#===============================================================================

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo ""

PRISM_RATE=$(echo "scale=1; $PRISM_DETECTED * 100 / $TOTAL" | bc)
FPOCKET_RATE=$(echo "scale=1; $FPOCKET_DETECTED * 100 / $TOTAL" | bc)

echo -e "${CYAN}RESULTS SUMMARY${NC}"
echo ""
printf "%-20s %s\n" "Total structures:" "$TOTAL"
printf "%-20s %s\n" "Detection threshold:" "${DETECTION_THRESHOLD} (50% residue overlap)"
echo ""
printf "%-20s ${GREEN}%d/%d (%.1f%%)${NC}\n" "PRISM detected:" "$PRISM_DETECTED" "$TOTAL" "$PRISM_RATE"
printf "%-20s ${BLUE}%d/%d (%.1f%%)${NC}\n" "fpocket detected:" "$FPOCKET_DETECTED" "$TOTAL" "$FPOCKET_RATE"
echo ""

if (( $(echo "$PRISM_RATE > $FPOCKET_RATE" | bc -l) )); then
    DIFF=$(echo "$PRISM_RATE - $FPOCKET_RATE" | bc)
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  🎉 PRISM beats fpocket by ${DIFF}% on cryptic site detection!      ${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════╝${NC}"
    
    if (( $(echo "$DIFF >= 10" | bc -l) )); then
        echo ""
        echo -e "${GREEN}This is a PUBLISHABLE RESULT. Consider submitting to:${NC}"
        echo "  - Journal of Chemical Information and Modeling"
        echo "  - Bioinformatics"
        echo "  - PLOS Computational Biology"
    fi
elif (( $(echo "$FPOCKET_RATE > $PRISM_RATE" | bc -l) )); then
    DIFF=$(echo "$FPOCKET_RATE - $PRISM_RATE" | bc)
    echo -e "${YELLOW}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${YELLOW}║  fpocket wins by ${DIFF}%. Algorithm improvements needed.          ${NC}"
    echo -e "${YELLOW}╚══════════════════════════════════════════════════════════════════╝${NC}"
else
    echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║  TIE - Both tools have same detection rate.                      ${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════╝${NC}"
fi

# Save results to CSV
RESULTS_CSV="$RESULTS_DIR/benchmark_results.csv"
echo "tool,detected,total,rate" > "$RESULTS_CSV"
echo "prism,$PRISM_DETECTED,$TOTAL,$PRISM_RATE" >> "$RESULTS_CSV"
echo "fpocket,$FPOCKET_DETECTED,$TOTAL,$FPOCKET_RATE" >> "$RESULTS_CSV"
echo ""
echo "Results saved to: $RESULTS_CSV"
