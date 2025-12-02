#!/bin/bash
# ============================================================================
# PRISM-LBS Pocket Detection Diagnostic v1.0
#
# Now that residue indexing is fixed, this analyzes WHY certain binding sites
# aren't being detected as cohesive pockets.
# ============================================================================

set -e

PRISM_BINARY="${PRISM_BINARY:-./target/release/prism-lbs}"
DIAG_DIR="/tmp/prism_pocket_diag"
mkdir -p "$DIAG_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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
    echo -e "${MAGENTA}═══════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${MAGENTA}  ${BOLD}$1${NC}"
    echo -e "${MAGENTA}═══════════════════════════════════════════════════════════════════════════════${NC}"
}

# ============================================================================
# STRUCTURES TO ANALYZE
# ============================================================================

# Passing structures (for comparison)
declare -A PASSING=(
    ["1hpv"]="25;27;29;30;48;49;50;51;52;53;80;81;82;84"
    ["1hsg"]="25;27;29;30;48;49;50;51;52;53;80;81;82;84"
    ["4hvp"]="25;27;29;30;48;49;50;51;52;53;80;81;82;84"
)

# Failing structures (need investigation)
declare -A FAILING=(
    ["1m17"]="695;696;697;718;719;720;721;726;743;744;790;791;792;793;855"
    ["2rh1"]="113;114;117;118;193;203;204;207;286;289;290;293;312"
    ["3eml"]="10;13;18;31;33;80;81;82;83;84;85;86;131;144;145"
    ["3ptb"]="57;102;189;190;191;192;195;213;214;215;216;217"
    ["1opk"]="20;21;48;110;111;112;298;299;300;301;302;303"
    ["3nup"]="62;63;64;91;92;94;96;117;118;119;121;198;199;200;201;209"
)

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

download_pdb() {
    local pdb_id=$1
    local pdb_file="${DIAG_DIR}/${pdb_id}.pdb"

    if [[ ! -f "$pdb_file" ]]; then
        wget -q "https://files.rcsb.org/download/${pdb_id^^}.pdb" -O "$pdb_file" 2>/dev/null || return 1
    fi
    echo "$pdb_file"
}

analyze_binding_site_geometry() {
    local pdb_file=$1
    local gt_residues=$2

    echo "  Binding site geometry analysis:"

    # Extract CA atoms for ground truth residues
    IFS=';' read -ra GT_ARR <<< "$gt_residues"

    local coords=""
    local found=0
    local total=${#GT_ARR[@]}

    for resnum in "${GT_ARR[@]}"; do
        # Get CA atom coordinates
        local ca_line=$(grep "^ATOM" "$pdb_file" | grep " CA " | awk -v r="$resnum" '$6 == r {print; exit}')
        if [[ -n "$ca_line" ]]; then
            local x=$(echo "$ca_line" | awk '{print $7}')
            local y=$(echo "$ca_line" | awk '{print $8}')
            local z=$(echo "$ca_line" | awk '{print $9}')
            coords="${coords}${x},${y},${z};"
            ((found++))
        fi
    done

    echo "    Residues found in PDB: ${found}/${total}"

    if [[ $found -lt 4 ]]; then
        echo -e "    ${RED}✗ Too few residues found - ground truth may be wrong${NC}"
        return
    fi

    # Calculate approximate site size (bounding box)
    echo "$coords" | tr ';' '\n' | grep -v '^$' > "${DIAG_DIR}/coords.tmp"

    if [[ -s "${DIAG_DIR}/coords.tmp" ]]; then
        local x_vals=$(cut -d',' -f1 "${DIAG_DIR}/coords.tmp")
        local y_vals=$(cut -d',' -f2 "${DIAG_DIR}/coords.tmp")
        local z_vals=$(cut -d',' -f3 "${DIAG_DIR}/coords.tmp")

        local x_min=$(echo "$x_vals" | sort -n | head -1)
        local x_max=$(echo "$x_vals" | sort -n | tail -1)
        local y_min=$(echo "$y_vals" | sort -n | head -1)
        local y_max=$(echo "$y_vals" | sort -n | tail -1)
        local z_min=$(echo "$z_vals" | sort -n | head -1)
        local z_max=$(echo "$z_vals" | sort -n | tail -1)

        local x_span=$(echo "$x_max - $x_min" | bc)
        local y_span=$(echo "$y_max - $y_min" | bc)
        local z_span=$(echo "$z_max - $z_min" | bc)

        echo "    Bounding box: ${x_span}Å × ${y_span}Å × ${z_span}Å"

        # Check if site is elongated (could be split across pockets)
        local max_span=$(echo -e "${x_span}\n${y_span}\n${z_span}" | sort -n | tail -1)
        local min_span=$(echo -e "${x_span}\n${y_span}\n${z_span}" | sort -n | head -1)
        local aspect=$(echo "scale=1; $max_span / ($min_span + 0.1)" | bc)

        if (( $(echo "$aspect > 3" | bc -l) )); then
            echo -e "    ${YELLOW}⚠ Elongated site (aspect ratio: ${aspect}) - may be split across pockets${NC}"
        fi

        if (( $(echo "$max_span > 25" | bc -l) )); then
            echo -e "    ${YELLOW}⚠ Large site (${max_span}Å span) - may exceed single pocket detection${NC}"
        fi
    fi
}

analyze_pocket_coverage() {
    local json_file=$1
    local gt_residues=$2
    local pdb_id=$3

    echo ""
    echo "  Pocket coverage analysis:"

    IFS=';' read -ra GT_ARR <<< "$gt_residues"
    local gt_count=${#GT_ARR[@]}

    # Check each pocket for overlap
    local pocket_count=$(jq '.pockets | length' "$json_file")
    echo "    Total pockets detected: $pocket_count"

    declare -A residue_coverage
    for res in "${GT_ARR[@]}"; do
        residue_coverage[$res]=0
    done

    echo ""
    echo "    Pocket-by-pocket analysis:"

    for ((i=0; i<pocket_count && i<10; i++)); do
        local pocket_residues=$(jq -r ".pockets[$i].residue_indices[]" "$json_file" 2>/dev/null | tr '\n' ' ')
        local pocket_volume=$(jq ".pockets[$i].volume // 0" "$json_file")
        local pocket_drug=$(jq ".pockets[$i].druggability_score.total // 0" "$json_file")

        local overlap=0
        local matched_res=""
        for res in "${GT_ARR[@]}"; do
            if [[ " $pocket_residues " =~ " $res " ]]; then
                ((overlap++))
                matched_res="${matched_res}${res},"
                residue_coverage[$res]=1
            fi
        done

        local overlap_pct=0
        [[ $gt_count -gt 0 ]] && overlap_pct=$(echo "scale=1; $overlap * 100 / $gt_count" | bc)

        if [[ $overlap -gt 0 ]]; then
            echo "      Pocket $((i+1)): ${overlap}/${gt_count} (${overlap_pct}%) - Vol:${pocket_volume}Å³ Drug:${pocket_drug}"
            echo "        Matched: ${matched_res%,}"
        fi
    done

    # Show uncovered residues
    echo ""
    echo "    Ground truth residue coverage:"
    local covered=0
    local uncovered=""
    for res in "${GT_ARR[@]}"; do
        if [[ "${residue_coverage[$res]}" == "1" ]]; then
            ((covered++))
        else
            uncovered="${uncovered}${res},"
        fi
    done

    echo "      Covered: ${covered}/${gt_count}"
    if [[ -n "$uncovered" ]]; then
        echo -e "      ${YELLOW}Uncovered residues: ${uncovered%,}${NC}"
    fi

    # Calculate if residues are split across pockets
    local pockets_with_gt=0
    for ((i=0; i<pocket_count && i<10; i++)); do
        local pocket_residues=$(jq -r ".pockets[$i].residue_indices[]" "$json_file" 2>/dev/null | tr '\n' ' ')
        for res in "${GT_ARR[@]}"; do
            if [[ " $pocket_residues " =~ " $res " ]]; then
                ((pockets_with_gt++))
                break
            fi
        done
    done

    if [[ $pockets_with_gt -gt 1 ]]; then
        echo ""
        echo -e "    ${YELLOW}⚠ Binding site split across $pockets_with_gt pockets${NC}"
        echo "      This suggests clustering parameters may need tuning"
    fi
}

analyze_detection_parameters() {
    local json_file=$1

    echo ""
    echo "  Detection parameters (from top pocket):"

    local top_volume=$(jq '.pockets[0].volume // 0' "$json_file")
    local top_enclosure=$(jq '.pockets[0].enclosure_ratio // 0' "$json_file")
    local top_depth=$(jq '.pockets[0].mean_depth // 0' "$json_file")
    local top_flex=$(jq '.pockets[0].mean_flexibility // 0' "$json_file")
    local top_hydro=$(jq '.pockets[0].mean_hydrophobicity // 0' "$json_file")

    echo "    Volume: ${top_volume} Å³"
    echo "    Enclosure ratio: ${top_enclosure}"
    echo "    Mean depth: ${top_depth} Å"
    echo "    Mean flexibility: ${top_flex}"
    echo "    Mean hydrophobicity: ${top_hydro}"

    # Flag potential issues
    if (( $(echo "$top_volume < 200" | bc -l) )); then
        echo -e "    ${YELLOW}⚠ Low volume - site may be too small or exposed${NC}"
    fi

    if (( $(echo "$top_enclosure < 0.3" | bc -l) )); then
        echo -e "    ${YELLOW}⚠ Low enclosure - site may be surface-exposed${NC}"
    fi
}

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

print_header "PRISM-LBS POCKET DETECTION DIAGNOSTIC"

echo ""
echo "Now that residue indexing is fixed, we analyze WHY certain binding sites"
echo "aren't being detected as cohesive pockets."
echo ""

# First, analyze passing structures to understand what works
print_section "PASSING STRUCTURES (Reference)"

for pdb_id in "${!PASSING[@]}"; do
    echo ""
    echo -e "${GREEN}━━━ ${pdb_id^^} ━━━${NC}"

    pdb_file=$(download_pdb "$pdb_id")
    json_file="${DIAG_DIR}/${pdb_id}.json"

    if [[ ! -f "$pdb_file" ]]; then
        echo "  Could not download PDB"
        continue
    fi

    "$PRISM_BINARY" --input "$pdb_file" --output "$json_file" 2>/dev/null

    if [[ ! -f "$json_file" ]]; then
        echo "  PRISM failed to produce output"
        continue
    fi

    gt="${PASSING[$pdb_id]}"
    analyze_binding_site_geometry "$pdb_file" "$gt"
    analyze_pocket_coverage "$json_file" "$gt" "$pdb_id"
    analyze_detection_parameters "$json_file"
done

# Analyze failing structures
print_section "FAILING STRUCTURES (Investigation)"

for pdb_id in "${!FAILING[@]}"; do
    echo ""
    echo -e "${RED}━━━ ${pdb_id^^} ━━━${NC}"

    pdb_file=$(download_pdb "$pdb_id")
    json_file="${DIAG_DIR}/${pdb_id}.json"

    if [[ ! -f "$pdb_file" ]]; then
        echo "  Could not download PDB"
        continue
    fi

    "$PRISM_BINARY" --input "$pdb_file" --output "$json_file" 2>/dev/null

    if [[ ! -f "$json_file" ]]; then
        echo "  PRISM failed to produce output"
        continue
    fi

    gt="${FAILING[$pdb_id]}"
    analyze_binding_site_geometry "$pdb_file" "$gt"
    analyze_pocket_coverage "$json_file" "$gt" "$pdb_id"
    analyze_detection_parameters "$json_file"
done

# Summary
print_section "DIAGNOSTIC SUMMARY"

echo ""
echo "Common failure patterns:"
echo ""
echo "1. BINDING SITE FRAGMENTATION"
echo "   Ground truth residues are detected but split across multiple pockets."
echo "   Solution: Tune clustering parameters (DBSCAN eps, min_samples)"
echo ""
echo "2. SURFACE-EXPOSED SITES"
echo "   Sites with low enclosure ratio may not register as 'pockets'."
echo "   Solution: Lower enclosure threshold or add surface site detection"
echo ""
echo "3. ELONGATED BINDING CHANNELS"
echo "   Long, narrow sites (like GPCR orthosteric sites) get fragmented."
echo "   Solution: Implement channel detection or adjust clustering"
echo ""
echo "4. LARGE KINASE ATP SITES"
echo "   ATP sites span 15+ residues across multiple structural elements."
echo "   Solution: Increase pocket merging radius or use structure-aware clustering"
echo ""

echo "Recommendations for world-class detection:"
echo ""
echo "  1. Implement pocket merging for adjacent cavities"
echo "  2. Add GPCR-specific detection mode"
echo "  3. Tune DBSCAN parameters based on protein class"
echo "  4. Consider HDBSCAN for variable-density clustering"
echo ""

# Cleanup
rm -f "${DIAG_DIR}/coords.tmp"

print_header "DIAGNOSTIC COMPLETE"
