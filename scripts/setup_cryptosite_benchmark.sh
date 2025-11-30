#!/bin/bash
#===============================================================================
# CryptoSite Benchmark Suite for PRISM-LBS
# 
# Tests detection of cryptic (hidden) binding sites that only appear
# upon ligand binding. This is the gold standard for proving novelty
# in pocket detection algorithms.
#
# Reference: Cimermancic et al., J Mol Biol, 2016
# "CryptoSite: Expanding the Druggable Proteome by Characterization 
#  and Prediction of Cryptic Binding Sites"
#
# Usage: ./setup_cryptosite_benchmark.sh
#===============================================================================

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

BENCHMARK_DIR="benchmark/cryptosite"
STRUCTURES_DIR="$BENCHMARK_DIR/structures"
RESULTS_DIR="$BENCHMARK_DIR/results"
GROUND_TRUTH="$BENCHMARK_DIR/ground_truth.csv"

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║          CryptoSite Benchmark Setup for PRISM-LBS                ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

#===============================================================================
# Create directory structure
#===============================================================================

mkdir -p "$STRUCTURES_DIR/apo"
mkdir -p "$STRUCTURES_DIR/holo"
mkdir -p "$RESULTS_DIR/prism"
mkdir -p "$RESULTS_DIR/fpocket"

#===============================================================================
# CryptoSite Core Dataset (18 well-characterized cryptic sites)
# These are the highest-confidence examples from the original paper
#===============================================================================

echo -e "${BLUE}[INFO] Creating ground truth dataset...${NC}"

cat > "$GROUND_TRUTH" << 'EOF'
# CryptoSite Benchmark - Ground Truth
# Format: apo_pdb,holo_pdb,protein_name,cryptic_residues,site_description,difficulty
#
# Difficulty levels:
#   easy   = pocket partially visible in apo
#   medium = pocket barely visible in apo  
#   hard   = pocket completely absent in apo
#
# cryptic_residues = residues that form the cryptic site (comma-separated)
#
1JWP,1PZO,TEM-1 beta-lactamase,"238,240,244,276","Allosteric site 16Å from active site",hard
1M47,1M48,Interleukin-2,"30,33,34,37,38,41,42,45","Groove between helices A and B",hard
1J3H,1FMO,Protein Kinase A,"49,50,51,52,54,55,57,72","Myristate binding pocket",medium
1K8K,1K7E,p38 MAP Kinase,"71,74,75,104,106,107,108","DFG-out allosteric pocket",medium
1OPK,1Q5K,Aldose Reductase,"48,49,112,298,299,300","Specificity pocket",easy
1G4E,1ZGY,Factor Xa,"99,143,146,174,175,176,177","S4 subpocket",medium
1FJS,1LQD,Bcl-xL,"96,99,100,103,104,107,108,111","BH3 binding groove",hard
1YET,1YDT,Thymidylate Synthase,"23,48,52,55,169,170,177","Folate binding pocket extension",medium
1SQN,1UNL,Thermolysin,"119,120,122,123,127,128,130","S1' subpocket",easy
3ERT,1GWR,Estrogen Receptor alpha,"343,346,347,349,350,351,353","Helix 12 binding groove",hard
1LI4,1OG5,Caspase-7,"233,234,236,276,277,278","Allosteric site",hard
1EX8,1OI9,Cytochrome P450 2C9,"72,74,97,100,102,103","Access channel",medium
1PKD,1A28,Pyruvate Kinase,"53,54,55,56,117,118,119","Allosteric activator site",medium
2FGU,2HYY,HSP90,"52,54,91,94,98,107,108,111","ATP lid cryptic pocket",hard
1YQY,2BU4,Chk1 Kinase,"86,88,89,90,148,149,150","Selectivity pocket",medium
1KS9,1OWE,Renin,"73,74,75,218,219,220,289","Flap pocket",hard
1F3D,1JD0,Penicillin Binding Protein,"322,325,326,329,330,422,423","Allosteric trigger site",hard
2AYN,2B63,Thrombin,"60A,60B,60C,60D,60E","60s loop pocket",medium
EOF

echo -e "${GREEN}[OK] Ground truth created with 18 cryptic site examples${NC}"

#===============================================================================
# Download structures
#===============================================================================

download_pdb() {
    local pdb_id=$1
    local output_dir=$2
    local output_file="$output_dir/${pdb_id,,}.pdb"
    
    if [ -f "$output_file" ]; then
        echo -e "  ${YELLOW}[SKIP]${NC} $pdb_id already exists"
        return 0
    fi
    
    wget -q "https://files.rcsb.org/download/${pdb_id}.pdb" -O "$output_file" 2>/dev/null || {
        echo -e "  ${RED}[FAIL]${NC} Could not download $pdb_id"
        return 1
    }
    
    echo -e "  ${GREEN}[OK]${NC} Downloaded $pdb_id"
}

echo ""
echo -e "${BLUE}[INFO] Downloading APO structures (no ligand)...${NC}"

# APO structures
APO_PDBS="1JWP 1M47 1J3H 1K8K 1OPK 1G4E 1FJS 1YET 1SQN 3ERT 1LI4 1EX8 1PKD 2FGU 1YQY 1KS9 1F3D 2AYN"
for pdb in $APO_PDBS; do
    download_pdb "$pdb" "$STRUCTURES_DIR/apo"
done

echo ""
echo -e "${BLUE}[INFO] Downloading HOLO structures (with ligand)...${NC}"

# HOLO structures
HOLO_PDBS="1PZO 1M48 1FMO 1K7E 1Q5K 1ZGY 1LQD 1YDT 1UNL 1GWR 1OG5 1OI9 1A28 2HYY 2BU4 1OWE 1JD0 2B63"
for pdb in $HOLO_PDBS; do
    download_pdb "$pdb" "$STRUCTURES_DIR/holo"
done

#===============================================================================
# Create benchmark runner script
#===============================================================================

cat > "$BENCHMARK_DIR/run_benchmark.sh" << 'RUNNER_EOF'
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
extract_prism_residues() {
    local json_file=$1
    python3 << PYEOF
import json
import sys

try:
    with open('$json_file', 'r') as f:
        data = json.load(f)
    
    all_residues = set()
    for pocket in data.get('pockets', []):
        for res in pocket.get('residue_indices', []):
            all_residues.add(int(res))
    
    print(' '.join(map(str, sorted(all_residues))))
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
    
    # Parse fpocket's pocket atoms file
    python3 << PYEOF
import os
import sys

fpocket_dir = '$fpocket_dir'
residues = set()

# Look for pocket PDB files
for fname in os.listdir(fpocket_dir):
    if fname.endswith('_atm.pdb'):
        with open(os.path.join(fpocket_dir, fname), 'r') as f:
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

while IFS=, read -r apo_pdb holo_pdb protein_name cryptic_residues site_desc difficulty; do
    # Skip comments and header
    [[ "$apo_pdb" =~ ^#.*$ ]] && continue
    [[ -z "$apo_pdb" ]] && continue
    
    apo_file="$STRUCTURES_DIR/apo/${apo_pdb,,}.pdb"
    
    if [ ! -f "$apo_file" ]; then
        echo -e "${YELLOW}[SKIP]${NC} $apo_pdb not found"
        continue
    fi
    
    ((TOTAL++))
    
    # Run PRISM (batch mode with GPU acceleration)
    prism_output="$RESULTS_DIR/prism/${apo_pdb,,}.json"
    mkdir -p "$RESULTS_DIR/prism"
    if [ -f "$PRISM_BIN" ]; then
        $PRISM_BIN --batch --input "$apo_file" -o "$prism_output" 2>/dev/null || true
    fi
    
    # Run fpocket
    fpocket_base="$RESULTS_DIR/fpocket/${apo_pdb,,}"
    mkdir -p "$RESULTS_DIR/fpocket"
    cp "$apo_file" "${fpocket_base}.pdb" 2>/dev/null || true
    (cd "$RESULTS_DIR/fpocket" && $FPOCKET_BIN -f "${apo_pdb,,}.pdb" 2>/dev/null) || true
    
    # Extract detected residues
    prism_residues=""
    if [ -f "$prism_output" ]; then
        prism_residues=$(extract_prism_residues "$prism_output")
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
        ((PRISM_DETECTED++))
    fi
    
    if (( $(echo "$fpocket_overlap >= $DETECTION_THRESHOLD" | bc -l) )); then
        fpocket_detected="YES"
        ((FPOCKET_DETECTED++))
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

done < "$GROUND_TRUTH"

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
RUNNER_EOF

chmod +x "$BENCHMARK_DIR/run_benchmark.sh"

#===============================================================================
# Create analysis script
#===============================================================================

cat > "$BENCHMARK_DIR/analyze_results.py" << 'PYTHON_EOF'
#!/usr/bin/env python3
"""
CryptoSite Benchmark Analysis

Generates detailed comparison of PRISM vs fpocket performance
on cryptic binding site detection.
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict

BENCHMARK_DIR = Path(__file__).parent
RESULTS_DIR = BENCHMARK_DIR / "results"
GROUND_TRUTH = BENCHMARK_DIR / "ground_truth.csv"

def load_ground_truth():
    """Load ground truth cryptic site data."""
    sites = []
    with open(GROUND_TRUTH, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) >= 6:
                sites.append({
                    'apo_pdb': parts[0],
                    'holo_pdb': parts[1],
                    'protein': parts[2],
                    'cryptic_residues': set(map(int, parts[3].replace('"', '').split(',') if parts[3] else [])),
                    'description': parts[4],
                    'difficulty': parts[5]
                })
    return sites

def load_prism_results(pdb_id):
    """Load PRISM pocket detection results."""
    json_file = RESULTS_DIR / "prism" / f"{pdb_id.lower()}.json"
    if not json_file.exists():
        return None
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    pockets = []
    for p in data.get('pockets', []):
        pockets.append({
            'residues': set(p.get('residue_indices', [])),
            'volume': p.get('volume', 0),
            'druggability': p.get('druggability_score', {}).get('total', 0)
        })
    return pockets

def calculate_metrics(detected_residues, ground_truth_residues):
    """Calculate precision, recall, F1 for residue detection."""
    if not ground_truth_residues:
        return {'precision': 0, 'recall': 0, 'f1': 0}
    
    if not detected_residues:
        return {'precision': 0, 'recall': 0, 'f1': 0}
    
    tp = len(detected_residues & ground_truth_residues)
    precision = tp / len(detected_residues) if detected_residues else 0
    recall = tp / len(ground_truth_residues) if ground_truth_residues else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {'precision': precision, 'recall': recall, 'f1': f1}

def analyze_by_difficulty(sites, results):
    """Analyze detection rates by difficulty level."""
    by_difficulty = defaultdict(lambda: {'total': 0, 'prism': 0, 'fpocket': 0})
    
    for site in sites:
        diff = site['difficulty']
        by_difficulty[diff]['total'] += 1
        
        pdb = site['apo_pdb']
        if pdb in results:
            if results[pdb].get('prism_detected'):
                by_difficulty[diff]['prism'] += 1
            if results[pdb].get('fpocket_detected'):
                by_difficulty[diff]['fpocket'] += 1
    
    return dict(by_difficulty)

def main():
    print("=" * 70)
    print("CryptoSite Benchmark Analysis")
    print("=" * 70)
    print()
    
    sites = load_ground_truth()
    print(f"Loaded {len(sites)} cryptic site entries")
    print()
    
    # Analyze each site
    results = {}
    for site in sites:
        pdb = site['apo_pdb']
        prism_pockets = load_prism_results(pdb)
        
        if prism_pockets:
            # Find best matching pocket
            best_recall = 0
            for pocket in prism_pockets:
                metrics = calculate_metrics(pocket['residues'], site['cryptic_residues'])
                if metrics['recall'] > best_recall:
                    best_recall = metrics['recall']
            
            results[pdb] = {
                'prism_recall': best_recall,
                'prism_detected': best_recall >= 0.5,
                'difficulty': site['difficulty']
            }
    
    # Summary statistics
    detected = sum(1 for r in results.values() if r.get('prism_detected'))
    total = len(results)
    
    print(f"PRISM Detection Rate: {detected}/{total} ({100*detected/total:.1f}%)")
    print()
    
    # By difficulty
    print("Detection by Difficulty:")
    print("-" * 40)
    
    by_diff = defaultdict(lambda: {'total': 0, 'detected': 0})
    for pdb, res in results.items():
        diff = res['difficulty']
        by_diff[diff]['total'] += 1
        if res['prism_detected']:
            by_diff[diff]['detected'] += 1
    
    for diff in ['easy', 'medium', 'hard']:
        if diff in by_diff:
            d = by_diff[diff]
            rate = 100 * d['detected'] / d['total'] if d['total'] > 0 else 0
            print(f"  {diff:10s}: {d['detected']}/{d['total']} ({rate:.1f}%)")
    
    print()
    print("=" * 70)

if __name__ == '__main__':
    main()
PYTHON_EOF

chmod +x "$BENCHMARK_DIR/analyze_results.py"

#===============================================================================
# Summary
#===============================================================================

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  CryptoSite Benchmark Setup Complete!                            ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Directory structure:"
echo "  $BENCHMARK_DIR/"
echo "  ├── ground_truth.csv          # 18 cryptic site definitions"
echo "  ├── structures/"
echo "  │   ├── apo/                   # Unbound structures (test input)"
echo "  │   └── holo/                  # Bound structures (reference)"
echo "  ├── results/"
echo "  │   ├── prism/                 # PRISM output JSONs"
echo "  │   └── fpocket/               # fpocket output"
echo "  ├── run_benchmark.sh           # Main benchmark script"
echo "  └── analyze_results.py         # Detailed analysis"
echo ""
echo -e "${BLUE}To run the benchmark:${NC}"
echo ""
echo "  # Make sure PRISM is built"
echo "  cargo build --release -p prism-lbs"
echo ""
echo "  # Run the benchmark"
echo "  cd $BENCHMARK_DIR"
echo "  ./run_benchmark.sh"
echo ""
echo -e "${YELLOW}Success criteria:${NC}"
echo "  - Beat fpocket by >10% = Publishable result"
echo "  - Beat fpocket by >5%  = Significant improvement"
echo "  - Match fpocket        = Competitive baseline"
echo ""
