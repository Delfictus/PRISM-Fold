#!/bin/bash
# ============================================================================
# PRISM-LBS Validation Diagnostic v1.0
# Identifies exactly why detection rates are low
# ============================================================================

set -e

PRISM_BINARY="${PRISM_BINARY:-./target/release/prism-lbs}"
BENCHMARK_DIR="benchmark/complete_validation"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m'

print_header() {
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC}  ${BOLD}$1${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"
}

print_section() {
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  ${BOLD}$1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════════════${NC}"
}

# ============================================================================
# DIAGNOSTIC 1: Check PRISM Binary and Version
# ============================================================================

print_header "PRISM-LBS VALIDATION DIAGNOSTIC"

print_section "1. PRISM Binary Check"

echo "Looking for PRISM binary..."
if [[ -f "$PRISM_BINARY" ]]; then
    echo -e "${GREEN}✓ Found: $PRISM_BINARY${NC}"
else
    # Try to find it
    for path in "./target/release/prism-lbs" "../target/release/prism-lbs" "prism-lbs" "$(which prism-lbs 2>/dev/null)"; do
        if [[ -f "$path" ]] || command -v "$path" &>/dev/null 2>&1; then
            PRISM_BINARY="$path"
            echo -e "${GREEN}✓ Found: $PRISM_BINARY${NC}"
            break
        fi
    done
fi

echo ""
echo "Version info:"
"$PRISM_BINARY" --version 2>&1 || echo "(version command not available)"

echo ""
echo "Help/usage:"
"$PRISM_BINARY" --help 2>&1 | head -30 || echo "(help not available)"

# ============================================================================
# DIAGNOSTIC 2: Test Single Structure Output Format
# ============================================================================

print_section "2. Output Format Analysis"

# Use 4HVP as test case (HIV protease - well characterized)
TEST_PDB="${BENCHMARK_DIR}/structures/tier1/4hvp.pdb"
TEST_OUTPUT="/tmp/prism_diagnostic_4hvp.json"

if [[ ! -f "$TEST_PDB" ]]; then
    echo "Downloading 4HVP for testing..."
    mkdir -p "${BENCHMARK_DIR}/structures/tier1"
    wget -q "https://files.rcsb.org/download/4HVP.pdb" -O "$TEST_PDB" || {
        echo -e "${RED}Failed to download 4HVP${NC}"
        exit 1
    }
fi

echo "Running PRISM on 4HVP..."
echo "Command: $PRISM_BINARY $TEST_PDB -o $TEST_OUTPUT"
"$PRISM_BINARY" "$TEST_PDB" -o "$TEST_OUTPUT" 2>&1 | head -20

if [[ -f "$TEST_OUTPUT" ]]; then
    echo -e "${GREEN}✓ Output file created${NC}"
    echo ""
    echo "File size: $(wc -c < "$TEST_OUTPUT") bytes"
    echo ""
    
    # Check JSON structure
    echo "Top-level keys:"
    jq -r 'keys[]' "$TEST_OUTPUT" 2>/dev/null || echo "Not valid JSON"
    
    echo ""
    echo "Number of pockets: $(jq '.pockets | length' "$TEST_OUTPUT" 2>/dev/null || echo "N/A")"
    
    echo ""
    echo "First pocket structure (keys):"
    jq -r '.pockets[0] | keys[]' "$TEST_OUTPUT" 2>/dev/null | head -20
    
    echo ""
    echo "First pocket sample data:"
    jq '.pockets[0] | {
        residue_indices: .residue_indices[:10],
        volume: .volume,
        druggability_total: .druggability_score.total,
        druggability_class: .druggability_score.classification,
        mean_flexibility: .mean_flexibility,
        mean_depth: .mean_depth
    }' "$TEST_OUTPUT" 2>/dev/null
    
else
    echo -e "${RED}✗ No output file created${NC}"
    echo "Checking stderr..."
    "$PRISM_BINARY" "$TEST_PDB" -o "$TEST_OUTPUT" 2>&1
fi

# ============================================================================
# DIAGNOSTIC 3: Residue Numbering Analysis
# ============================================================================

print_section "3. Residue Numbering Analysis"

echo "Ground truth residues for 4HVP (HIV-1 Protease active site):"
echo "  PDB Author numbering: 25, 27, 29, 30, 48, 49, 50, 51, 52, 53, 80, 81, 82, 84"
echo ""

echo "PRISM detected residues (all pockets):"
ALL_RESIDUES=$(jq -r '[.pockets[].residue_indices[]] | unique | sort | .[]' "$TEST_OUTPUT" 2>/dev/null | tr '\n' ' ')
echo "  $ALL_RESIDUES"
echo ""

echo "PRISM residue range:"
MIN_RES=$(jq '[.pockets[].residue_indices[]] | min' "$TEST_OUTPUT" 2>/dev/null)
MAX_RES=$(jq '[.pockets[].residue_indices[]] | max' "$TEST_OUTPUT" 2>/dev/null)
echo "  Min: $MIN_RES, Max: $MAX_RES"
echo ""

# Check PDB file residue numbering
echo "PDB file residue numbering (from ATOM records):"
grep "^ATOM" "$TEST_PDB" | awk '{print $6}' | sort -n | uniq | head -20 | tr '\n' ' '
echo "..."
echo ""

PDB_MIN=$(grep "^ATOM" "$TEST_PDB" | awk '{print $6}' | sort -n | uniq | head -1)
PDB_MAX=$(grep "^ATOM" "$TEST_PDB" | awk '{print $6}' | sort -n | uniq | tail -1)
echo "  PDB residue range: $PDB_MIN to $PDB_MAX"

# ============================================================================
# DIAGNOSTIC 4: Indexing Offset Analysis
# ============================================================================

print_section "4. Indexing Offset Analysis"

echo "Hypothesis: PRISM uses 0-indexed residues, ground truth uses PDB numbering"
echo ""

echo "If PRISM residue_indices are 0-indexed:"
echo "  Ground truth residue 25 → PRISM index 24 (or different based on first residue)"
echo ""

# Get first residue number from PDB
FIRST_RESIDUE=$(grep "^ATOM" "$TEST_PDB" | awk '{print $6}' | head -1)
echo "First residue in PDB: $FIRST_RESIDUE"
echo ""

# Calculate offset
if [[ "$MIN_RES" == "0" ]]; then
    echo -e "${YELLOW}PRISM appears to use 0-based indexing!${NC}"
    echo "Ground truth residue N → PRISM index (N - $FIRST_RESIDUE)"
    OFFSET=$FIRST_RESIDUE
else
    echo "PRISM may use 1-based or PDB numbering"
    OFFSET=0
fi

# ============================================================================
# DIAGNOSTIC 5: Pocket-by-Pocket Analysis for 4HVP
# ============================================================================

print_section "5. 4HVP Pocket Analysis"

GT_RESIDUES="25;27;29;30;48;49;50;51;52;53;80;81;82;84"
echo "Ground truth active site residues: $GT_RESIDUES"
echo ""

# Convert to array
IFS=';' read -ra GT_ARRAY <<< "$GT_RESIDUES"

echo "Checking each pocket for overlap with ground truth:"
echo ""

POCKET_COUNT=$(jq '.pockets | length' "$TEST_OUTPUT")
for ((i=0; i<POCKET_COUNT && i<10; i++)); do
    POCKET_RESIDUES=$(jq -r ".pockets[$i].residue_indices | .[]" "$TEST_OUTPUT" | tr '\n' ' ')
    POCKET_VOLUME=$(jq ".pockets[$i].volume" "$TEST_OUTPUT")
    POCKET_DRUG=$(jq ".pockets[$i].druggability_score.total" "$TEST_OUTPUT")
    
    # Count direct matches (no offset)
    DIRECT_MATCH=0
    for gt in "${GT_ARRAY[@]}"; do
        if [[ " $POCKET_RESIDUES " =~ " $gt " ]]; then
            ((DIRECT_MATCH++))
        fi
    done
    
    # Count offset matches (GT - offset)
    OFFSET_MATCH=0
    for gt in "${GT_ARRAY[@]}"; do
        offset_idx=$((gt - FIRST_RESIDUE))
        if [[ " $POCKET_RESIDUES " =~ " $offset_idx " ]]; then
            ((OFFSET_MATCH++))
        fi
    done
    
    echo "Pocket $((i+1)): Volume=${POCKET_VOLUME}Å³, Drug=${POCKET_DRUG}"
    echo "  Residues: ${POCKET_RESIDUES:0:60}..."
    echo "  Direct matches: $DIRECT_MATCH/${#GT_ARRAY[@]} ($(echo "scale=1; $DIRECT_MATCH * 100 / ${#GT_ARRAY[@]}" | bc)%)"
    echo "  Offset matches: $OFFSET_MATCH/${#GT_ARRAY[@]} ($(echo "scale=1; $OFFSET_MATCH * 100 / ${#GT_ARRAY[@]}" | bc)%)"
    echo ""
done

# ============================================================================
# DIAGNOSTIC 6: Module Activation Check
# ============================================================================

print_section "6. Module Activation Check"

echo "Checking which detection modules are active..."
echo ""

# Look for detection_type or similar fields
echo "Detection types in output:"
jq -r '.pockets[].detection_type // "NOT_PRESENT"' "$TEST_OUTPUT" 2>/dev/null | sort | uniq -c

echo ""
echo "Looking for module indicators in PRISM help:"
"$PRISM_BINARY" --help 2>&1 | grep -iE "(module|detect|geometric|cryptic|allosteric|unified|consensus)" | head -10 || echo "No module flags found in help"

echo ""
echo "Checking for flags to enable specific modules:"
"$PRISM_BINARY" --help 2>&1 | grep -iE "(-m|--mode|--type|--module)" | head -5 || echo "No module flags found"

# ============================================================================
# DIAGNOSTIC 7: Test with Different Flags
# ============================================================================

print_section "7. Testing Different Invocation Modes"

echo "Testing various command line options..."
echo ""

# Test 1: Basic
echo "Test 1: Basic invocation"
echo "  Command: $PRISM_BINARY $TEST_PDB -o /tmp/test1.json"
"$PRISM_BINARY" "$TEST_PDB" -o /tmp/test1.json 2>&1 | head -5
POCKETS_1=$(jq '.pockets | length' /tmp/test1.json 2>/dev/null || echo "0")
echo "  Result: $POCKETS_1 pockets"
echo ""

# Test 2: With verbose
echo "Test 2: Verbose mode"
echo "  Command: $PRISM_BINARY $TEST_PDB -o /tmp/test2.json -v"
"$PRISM_BINARY" "$TEST_PDB" -o /tmp/test2.json -v 2>&1 | head -10 || echo "  -v flag not supported"
echo ""

# Test 3: Check for mode flags
echo "Test 3: Checking available subcommands/modes"
"$PRISM_BINARY" 2>&1 | head -20 || true

# ============================================================================
# DIAGNOSTIC 8: Summary and Recommendations
# ============================================================================

print_section "8. DIAGNOSTIC SUMMARY"

echo ""
echo "Findings:"
echo "─────────────────────────────────────────────────────────────────────────────"

if [[ "$MIN_RES" == "0" ]]; then
    echo -e "${YELLOW}⚠ CRITICAL: PRISM uses 0-based residue indexing${NC}"
    echo "  Ground truth uses PDB author numbering (1-based with possible gaps)"
    echo "  This explains the low overlap scores!"
    echo ""
    echo "  SOLUTION: Modify validation suite to convert ground truth residues:"
    echo "    ground_truth_idx = pdb_residue_number - first_residue_in_pdb"
    ISSUE_FOUND="indexing"
else
    echo "  PRISM residue indexing appears to be: $MIN_RES to $MAX_RES"
fi

echo ""
echo "PRISM output format:"
if jq -e '.pockets[0].detection_type' "$TEST_OUTPUT" >/dev/null 2>&1; then
    echo -e "${GREEN}✓ detection_type field present${NC}"
else
    echo -e "${YELLOW}⚠ detection_type field NOT present${NC}"
    echo "  The unified/consensus module may not be active"
    echo "  Check if --mode or similar flag is needed"
fi

echo ""
echo "Recommendations:"
echo "─────────────────────────────────────────────────────────────────────────────"

if [[ "${ISSUE_FOUND:-}" == "indexing" ]]; then
    echo "1. FIX REQUIRED: Update validation suite to handle 0-based indexing"
    echo "   OR update PRISM to output PDB residue numbers"
    echo ""
    echo "2. The overlap calculation needs to map:"
    echo "   PDB residue number → PRISM internal index"
    echo "   Formula: prism_idx = pdb_resnum - first_pdb_resnum"
fi

echo ""
echo "─────────────────────────────────────────────────────────────────────────────"

# ============================================================================
# DIAGNOSTIC 9: Create Corrected Ground Truth
# ============================================================================

print_section "9. Creating Corrected Ground Truth Mapping"

echo "To fix the validation suite, we need to map PDB residue numbers"
echo "to PRISM's internal 0-based indices."
echo ""
echo "For 4HVP (first residue = $FIRST_RESIDUE):"
echo ""
echo "Original ground truth: $GT_RESIDUES"
echo "Corrected (0-based):   "

CORRECTED=""
for gt in "${GT_ARRAY[@]}"; do
    idx=$((gt - FIRST_RESIDUE))
    CORRECTED="${CORRECTED}${idx};"
done
CORRECTED="${CORRECTED%;}"
echo "  $CORRECTED"

echo ""
echo "Verification - checking overlap with corrected indices:"
CORRECTED_MATCH=0
IFS=';' read -ra CORR_ARRAY <<< "$CORRECTED"
TOP_POCKET_RESIDUES=$(jq -r '.pockets[0].residue_indices | .[]' "$TEST_OUTPUT" | tr '\n' ' ')

for idx in "${CORR_ARRAY[@]}"; do
    if [[ " $TOP_POCKET_RESIDUES " =~ " $idx " ]]; then
        ((CORRECTED_MATCH++))
    fi
done

CORRECTED_PCT=$(echo "scale=1; $CORRECTED_MATCH * 100 / ${#CORR_ARRAY[@]}" | bc)
echo "Top pocket overlap with corrected indices: $CORRECTED_MATCH/${#CORR_ARRAY[@]} (${CORRECTED_PCT}%)"

if (( $(echo "$CORRECTED_PCT > 40" | bc -l) )); then
    echo -e "${GREEN}✓ CONFIRMED: 0-based indexing fix improves overlap significantly!${NC}"
else
    echo -e "${YELLOW}⚠ Issue may be more complex than simple indexing offset${NC}"
fi

echo ""
print_header "DIAGNOSTIC COMPLETE"
