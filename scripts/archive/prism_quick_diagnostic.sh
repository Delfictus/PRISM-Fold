#!/bin/bash
# ============================================================================
# PRISM-LBS Quick Diagnostic
# Identifies residue indexing issues and module activation
# ============================================================================

set -e

PRISM_BINARY="${PRISM_BINARY:-./target/release/prism-lbs}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${NC}  ${BOLD}PRISM-LBS QUICK DIAGNOSTIC${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Find PRISM binary
if [[ ! -f "$PRISM_BINARY" ]]; then
    for path in "./target/release/prism-lbs" "../target/release/prism-lbs" "prism-lbs"; do
        if [[ -f "$path" ]] || command -v "$path" &>/dev/null 2>&1; then
            PRISM_BINARY="$path"
            break
        fi
    done
fi

echo -e "${BOLD}1. PRISM Binary${NC}"
echo "   Location: $PRISM_BINARY"
if [[ -f "$PRISM_BINARY" ]]; then
    echo -e "   Status: ${GREEN}Found${NC}"
    "$PRISM_BINARY" --version 2>&1 | head -3 || true
else
    echo -e "   Status: ${RED}NOT FOUND${NC}"
    exit 1
fi

# Download test structure
echo ""
echo -e "${BOLD}2. Test Structure (4HVP - HIV Protease)${NC}"
mkdir -p /tmp/prism_diag
TEST_PDB="/tmp/prism_diag/4hvp.pdb"
TEST_JSON="/tmp/prism_diag/4hvp.json"

if [[ ! -f "$TEST_PDB" ]]; then
    echo "   Downloading..."
    wget -q "https://files.rcsb.org/download/4HVP.pdb" -O "$TEST_PDB"
fi

# Analyze PDB
echo "   PDB file: $TEST_PDB"
FIRST_RES=$(grep "^ATOM" "$TEST_PDB" | awk '{print $6}' | head -1)
LAST_RES=$(grep "^ATOM" "$TEST_PDB" | awk '{print $6}' | tail -1)
UNIQUE_RES=$(grep "^ATOM" "$TEST_PDB" | awk '{print $6}' | sort -nu | wc -l)
echo "   PDB residue range: $FIRST_RES to $LAST_RES ($UNIQUE_RES unique residues)"

# Run PRISM
echo ""
echo -e "${BOLD}3. Running PRISM${NC}"
echo "   Command: PRISM_PTX_DIR=target/ptx $PRISM_BINARY -i $TEST_PDB -o $TEST_JSON --unified"
PRISM_PTX_DIR="$(dirname "$PRISM_BINARY")/../target/ptx" "$PRISM_BINARY" -i "$TEST_PDB" -o "$TEST_JSON" --unified 2>&1 | head -5 || true

if [[ ! -f "$TEST_JSON" ]]; then
    echo -e "   ${RED}ERROR: No output produced${NC}"
    exit 1
fi

# Analyze output
echo ""
echo -e "${BOLD}4. PRISM Output Analysis${NC}"
POCKET_COUNT=$(jq '.pockets | length' "$TEST_JSON")
echo "   Pockets detected: $POCKET_COUNT"

PRISM_MIN=$(jq '[.pockets[].residue_indices[]] | min' "$TEST_JSON")
PRISM_MAX=$(jq '[.pockets[].residue_indices[]] | max' "$TEST_JSON")
echo "   PRISM residue index range: $PRISM_MIN to $PRISM_MAX"

# Check indexing
echo ""
echo -e "${BOLD}5. Indexing Analysis${NC}"
if [[ "$PRISM_MIN" == "0" ]]; then
    echo -e "   ${YELLOW}PRISM uses 0-BASED indexing${NC}"
    echo "   Ground truth PDB residue N → PRISM index (N - $FIRST_RES)"
    INDEXING="zero"
else
    echo "   PRISM may use 1-based or PDB numbering"
    INDEXING="other"
fi

# Test overlap with ground truth
echo ""
echo -e "${BOLD}6. Ground Truth Overlap Test${NC}"

# HIV Protease active site (PDB numbering)
GT_PDB="25;27;29;30;48;49;50;51;52;53;80;81;82;84"
echo "   Ground truth (PDB): $GT_PDB"

# Convert if needed
if [[ "$INDEXING" == "zero" ]]; then
    # Build residue mapping
    declare -A RESMAP
    idx=0
    prev=""
    while read -r resnum; do
        if [[ "$resnum" != "$prev" ]]; then
            RESMAP[$resnum]=$idx
            ((idx++))
            prev=$resnum
        fi
    done < <(grep "^ATOM" "$TEST_PDB" | awk '{print $6}' | uniq)
    
    # Convert
    GT_PRISM=""
    IFS=';' read -ra GT_ARR <<< "$GT_PDB"
    for res in "${GT_ARR[@]}"; do
        if [[ -n "${RESMAP[$res]:-}" ]]; then
            [[ -n "$GT_PRISM" ]] && GT_PRISM="${GT_PRISM};"
            GT_PRISM="${GT_PRISM}${RESMAP[$res]}"
        fi
    done
    echo "   Ground truth (PRISM 0-based): $GT_PRISM"
fi

# Get top pocket residues
TOP_RESIDUES=$(jq -r '.pockets[0].residue_indices | .[]' "$TEST_JSON" | tr '\n' ' ')
echo "   Top pocket residues: ${TOP_RESIDUES:0:60}..."

# Calculate overlap
echo ""
echo -e "${BOLD}7. Overlap Calculation${NC}"

# Without conversion
echo "   Testing WITHOUT index conversion:"
overlap_direct=0
IFS=';' read -ra GT_ARR <<< "$GT_PDB"
for gt in "${GT_ARR[@]}"; do
    if [[ " $TOP_RESIDUES " =~ " $gt " ]]; then
        ((overlap_direct++))
    fi
done
pct_direct=$(echo "scale=1; $overlap_direct * 100 / ${#GT_ARR[@]}" | bc)
echo "   Direct match: $overlap_direct/${#GT_ARR[@]} (${pct_direct}%)"

# With conversion
if [[ -n "${GT_PRISM:-}" ]]; then
    echo ""
    echo "   Testing WITH index conversion:"
    overlap_converted=0
    IFS=';' read -ra GT_PRISM_ARR <<< "$GT_PRISM"
    for gt in "${GT_PRISM_ARR[@]}"; do
        if [[ " $TOP_RESIDUES " =~ " $gt " ]]; then
            ((overlap_converted++))
        fi
    done
    pct_converted=$(echo "scale=1; $overlap_converted * 100 / ${#GT_PRISM_ARR[@]}" | bc)
    echo "   Converted match: $overlap_converted/${#GT_PRISM_ARR[@]} (${pct_converted}%)"
    
    echo ""
    if (( $(echo "$pct_converted > $pct_direct" | bc -l) )); then
        echo -e "   ${GREEN}✓ INDEX CONVERSION IMPROVES OVERLAP${NC}"
        echo "   The validation suite needs to convert ground truth residues"
    else
        echo -e "   ${YELLOW}⚠ Conversion did not help - investigate further${NC}"
    fi
fi

# Check for detection_type field
echo ""
echo -e "${BOLD}8. Module Activation Check${NC}"
if jq -e '.pockets[0].detection_type' "$TEST_JSON" >/dev/null 2>&1; then
    DT=$(jq -r '.pockets[0].detection_type' "$TEST_JSON")
    echo -e "   detection_type: ${GREEN}$DT${NC}"
else
    echo -e "   detection_type: ${YELLOW}NOT PRESENT${NC}"
    echo "   (Unified/consensus detection may not be active)"
fi

# Check available fields
echo ""
echo "   Available pocket fields:"
jq -r '.pockets[0] | keys[]' "$TEST_JSON" | head -15 | while read field; do
    echo "     • $field"
done

# Check world-class modules in source
echo ""
echo -e "${BOLD}9. World-Class Module Check${NC}"

SRC_DIR="crates/prism-lbs/src"
if [[ -d "$SRC_DIR" ]]; then
    modules=(
        "pocket/delaunay_detector.rs:Delaunay Alpha Spheres"
        "softspot/lanczos.rs:Lanczos Eigensolver"
        "allosteric/gpu_floyd_warshall.rs:GPU Floyd-Warshall"
        "pocket/sasa.rs:Shrake-Rupley SASA"
        "pocket/hdbscan.rs:HDBSCAN Clustering"
    )
    
    for module in "${modules[@]}"; do
        IFS=':' read -r path name <<< "$module"
        if [[ -f "${SRC_DIR}/${path}" ]]; then
            lines=$(wc -l < "${SRC_DIR}/${path}")
            echo -e "   ${GREEN}✓ $name${NC} ($lines lines)"
        else
            echo -e "   ${RED}✗ $name${NC} (NOT FOUND)"
        fi
    done
else
    echo "   Source directory not found at: $SRC_DIR"
fi

# Summary
echo ""
echo -e "${CYAN}══════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}DIAGNOSTIC SUMMARY${NC}"
echo -e "${CYAN}══════════════════════════════════════════════════════════════════════════════${NC}"
echo ""

if [[ "$INDEXING" == "zero" ]]; then
    echo -e "  ${YELLOW}ISSUE IDENTIFIED: PRISM uses 0-based residue indexing${NC}"
    echo ""
    echo "  The validation suite was comparing:"
    echo "    • Ground truth: PDB residue numbers (e.g., 25, 27, 29...)"
    echo "    • PRISM output: 0-based indices (e.g., 24, 26, 28...)"
    echo ""
    echo "  SOLUTION: Use prism_validation_suite_v3_corrected.sh"
    echo "            which converts ground truth to PRISM indices"
fi

echo ""
echo "  Run the corrected validation suite:"
echo "    ./scripts/prism_validation_suite_v3_corrected.sh"
echo ""

# Cleanup
rm -f "$TEST_JSON"
