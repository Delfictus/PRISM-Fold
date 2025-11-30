#!/bin/bash
#===============================================================================
# PRISM-LBS Basic Validation Script (Tier 1)
#
# Tests detection of well-known binding sites to establish basic correctness.
# These are not novel discoveries - just sanity checks.
#
# Usage: ./scripts/validate_basic.sh
#===============================================================================

set -euo pipefail

# Export CUDA environment for GPU acceleration
export PATH="/home/diddy/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:/usr/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12.6
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:${LD_LIBRARY_PATH:-}

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
STRUCTURES_DIR="$PROJECT_DIR/benchmark/basic_structures"
RESULTS_DIR="$PROJECT_DIR/benchmark/basic_results"

# PRISM binary path (unified TUI with GPU acceleration)
PRISM_BIN="${PRISM_BIN:-$PROJECT_DIR/target/release/prism}"

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║          PRISM-LBS Basic Validation (Tier 1)                     ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

#===============================================================================
# Setup directories
#===============================================================================

mkdir -p "$STRUCTURES_DIR"
mkdir -p "$RESULTS_DIR"

#===============================================================================
# Download test structures
#===============================================================================

download_pdb() {
    local pdb_id=$1
    local output_file="$STRUCTURES_DIR/${pdb_id,,}.pdb"

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

echo -e "${BLUE}[INFO] Downloading test structures...${NC}"

# Test structures
download_pdb "4HVP"  # HIV-1 Protease
download_pdb "3PTB"  # Trypsin
download_pdb "4DFR"  # Dihydrofolate Reductase
download_pdb "1HPV"  # Alternate HIV-1 protease for cross-validation

echo ""

#===============================================================================
# Validation functions
#===============================================================================

check_residues_in_pocket() {
    local json_file=$1
    shift
    local required_residues=("$@")
    # Convert bash array to comma-separated string for Python
    local required_list=$(IFS=,; echo "${required_residues[*]}")

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

    required = [$required_list]
    found = sum(1 for r in required if r in all_residues)

    # Pass if at least 50% of required residues found
    threshold = len(required) * 0.5
    if found >= threshold:
        print(f"PASS ({found}/{len(required)} residues)")
        sys.exit(0)
    else:
        print(f"FAIL ({found}/{len(required)} residues)")
        sys.exit(1)

except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(2)
PYEOF
}

check_volume_range() {
    local json_file=$1
    local min_vol=$2
    local max_vol=$3

    python3 << PYEOF
import json
import sys

try:
    with open('$json_file', 'r') as f:
        data = json.load(f)

    pockets = data.get('pockets', [])
    if not pockets:
        print("FAIL (no pockets found)")
        sys.exit(1)

    # Check if any pocket has volume in expected range
    for pocket in pockets:
        vol = pocket.get('volume', 0)
        if $min_vol <= vol <= $max_vol:
            print(f"PASS (volume={vol:.1f} Å³)")
            sys.exit(0)

    vols = [p.get('volume', 0) for p in pockets]
    print(f"FAIL (volumes: {[f'{v:.0f}' for v in vols[:3]]}...)")
    sys.exit(1)

except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(2)
PYEOF
}

check_druggability() {
    local json_file=$1
    local min_score=$2

    python3 << PYEOF
import json
import sys

try:
    with open('$json_file', 'r') as f:
        data = json.load(f)

    pockets = data.get('pockets', [])
    if not pockets:
        print("FAIL (no pockets)")
        sys.exit(1)

    # Get top pocket druggability
    top = pockets[0]
    score = top.get('druggability_score', {})
    if isinstance(score, dict):
        total = score.get('total', 0)
    else:
        total = float(score)

    if total >= $min_score:
        print(f"PASS (druggability={total:.3f})")
        sys.exit(0)
    else:
        print(f"FAIL (druggability={total:.3f} < $min_score)")
        sys.exit(1)

except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(2)
PYEOF
}

#===============================================================================
# Run tests
#===============================================================================

PASSED=0
FAILED=0
TOTAL=0

run_test() {
    local name=$1
    local pdb=$2
    local test_func=$3
    shift 3
    local args=("$@")

    ((TOTAL++)) || true

    local pdb_file="$STRUCTURES_DIR/${pdb,,}.pdb"
    local result_file="$RESULTS_DIR/${pdb,,}.json"

    if [ ! -f "$pdb_file" ]; then
        echo -e "  ${RED}[SKIP]${NC} $name - structure not found"
        return
    fi

    # Run PRISM if result doesn't exist or is older than binary
    if [ ! -f "$result_file" ] || [ "$PRISM_BIN" -nt "$result_file" ]; then
        if [ -f "$PRISM_BIN" ]; then
            # Use --batch mode with --input and -o for output
            $PRISM_BIN --batch --input "$pdb_file" -o "$result_file" 2>/dev/null || true
        else
            # Fallback: try cargo run
            (cd "$PROJECT_DIR" && cargo run --release -- --batch --input "$pdb_file" -o "$result_file" 2>/dev/null) || true
        fi
    fi

    if [ ! -f "$result_file" ]; then
        echo -e "  ${RED}[FAIL]${NC} $name - no output generated"
        ((FAILED++)) || true
        return
    fi

    # Run the test
    result=$($test_func "$result_file" "${args[@]}" 2>&1) || true

    if [[ "$result" == PASS* ]]; then
        echo -e "  ${GREEN}[PASS]${NC} $name: $result"
        ((PASSED++)) || true
    else
        echo -e "  ${RED}[FAIL]${NC} $name: $result"
        ((FAILED++)) || true
    fi
}

echo -e "${BLUE}[INFO] Running basic validation tests...${NC}"
echo ""

# HIV-1 Protease Tests
echo -e "${CYAN}HIV-1 Protease (4HVP):${NC}"
run_test "Active site residues" "4HVP" check_residues_in_pocket 25 27 49 50 80 81 82 84
run_test "Volume range (400-1000 Å³)" "4HVP" check_volume_range 400 1000
run_test "Druggability (>0.5)" "4HVP" check_druggability 0.5
echo ""

# Trypsin Tests
echo -e "${CYAN}Trypsin (3PTB):${NC}"
run_test "S1 pocket residues" "3PTB" check_residues_in_pocket 189 190 195 214 215 216
run_test "Volume range (200-800 Å³)" "3PTB" check_volume_range 200 800
run_test "Druggability (>0.4)" "3PTB" check_druggability 0.4
echo ""

# DHFR Tests
echo -e "${CYAN}Dihydrofolate Reductase (4DFR):${NC}"
run_test "Folate binding residues" "4DFR" check_residues_in_pocket 5 6 7 27 31 54 57 94
run_test "Volume range (300-900 Å³)" "4DFR" check_volume_range 300 900
run_test "Druggability (>0.4)" "4DFR" check_druggability 0.4
echo ""

#===============================================================================
# Summary
#===============================================================================

echo "═══════════════════════════════════════════════════════════════════"
echo ""

PASS_RATE=$(echo "scale=1; $PASSED * 100 / $TOTAL" | bc)

if [ "$FAILED" -eq 0 ]; then
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  TIER 1 VALIDATION: ALL TESTS PASSED ($PASSED/$TOTAL)                  ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════╝${NC}"
    exit 0
elif [ "$PASSED" -ge $(($TOTAL / 2)) ]; then
    echo -e "${YELLOW}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${YELLOW}║  TIER 1 VALIDATION: PARTIAL PASS ($PASSED/$TOTAL tests, ${PASS_RATE}%)        ║${NC}"
    echo -e "${YELLOW}╚══════════════════════════════════════════════════════════════════╝${NC}"
    exit 1
else
    echo -e "${RED}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║  TIER 1 VALIDATION: FAILED ($PASSED/$TOTAL tests, ${PASS_RATE}%)               ║${NC}"
    echo -e "${RED}╚══════════════════════════════════════════════════════════════════╝${NC}"
    exit 2
fi
