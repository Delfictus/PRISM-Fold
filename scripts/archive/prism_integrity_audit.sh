#!/bin/bash
# ============================================================================
# PRISM-LBS COMPREHENSIVE INTEGRITY AUDIT
# ============================================================================
#
# Purpose: Verify that PRISM-LBS produces legitimate, explainable results
# without any hardcoded values, test data leakage, or shortcuts.
#
# This audit checks:
# 1. No hardcoded PDB IDs, residue lists, or benchmark answers
# 2. GPU modules are actually compiled and called
# 3. Detection algorithms produce structure-dependent output
# 4. Druggability scores are computed, not memorized
# 5. All claimed modules exist and are integrated
# 6. Results are reproducible and deterministic
# 7. Performance scales with input size (not constant time cheating)
#
# ============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRISM_ROOT="${SCRIPT_DIR}/.."
AUDIT_DIR="${PRISM_ROOT}/audit_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
AUDIT_LOG="${AUDIT_DIR}/audit_${TIMESTAMP}.log"

# Initialize
mkdir -p "${AUDIT_DIR}"

log() {
    echo -e "$1" | tee -a "${AUDIT_LOG}"
}

pass() {
    log "${GREEN}[PASS]${NC} $1"
}

fail() {
    log "${RED}[FAIL]${NC} $1"
    FAILURES=$((FAILURES + 1))
}

warn() {
    log "${YELLOW}[WARN]${NC} $1"
    WARNINGS=$((WARNINGS + 1))
}

info() {
    log "${CYAN}[INFO]${NC} $1"
}

FAILURES=0
WARNINGS=0

# ============================================================================
print_header() {
    log ""
    log "${CYAN}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
    log "${CYAN}║${NC}  ${BOLD}$1${NC}"
    log "${CYAN}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"
}

print_section() {
    log ""
    log "${CYAN}═══════════════════════════════════════════════════════════════════════════════${NC}"
    log "  ${BOLD}$1${NC}"
    log "${CYAN}═══════════════════════════════════════════════════════════════════════════════${NC}"
}

# ============================================================================
# AUDIT 1: HARDCODED VALUE DETECTION
# ============================================================================

audit_hardcoded_values() {
    print_header "AUDIT 1: HARDCODED VALUE DETECTION"

    log ""
    log "Searching for potential hardcoded benchmark answers..."
    log ""

    # Check for hardcoded PDB IDs
    info "Checking for hardcoded PDB IDs in source code..."

    BENCHMARK_PDBS="1hpv|1hsg|4hvp|4dfr|3ptb|3eml|1m17|1opk|2rh1|3nup|1jwp|1v4s|3hrf|4obe|1c9y"

    HARDCODED_PDBS=$(grep -rniE "(\"|\')($BENCHMARK_PDBS)(\"|\')|\b($BENCHMARK_PDBS)\b" \
        "${PRISM_ROOT}/crates" \
        --include="*.rs" \
        --exclude-dir=test \
        --exclude-dir=benches \
        2>/dev/null | grep -v "// test" | grep -v "#\[test\]" | head -20 || true)

    if [[ -n "$HARDCODED_PDBS" ]]; then
        fail "Found hardcoded benchmark PDB IDs in source:"
        echo "$HARDCODED_PDBS" | head -10
    else
        pass "No hardcoded benchmark PDB IDs found in source code"
    fi

    # Check for hardcoded residue lists
    info "Checking for hardcoded residue lists..."

    # Known ground truth patterns (e.g., HIV protease binding site)
    GT_PATTERNS="25.*27.*29.*48.*49.*50|64.*65.*66.*151.*152|113.*114.*117.*118"

    HARDCODED_RESIDUES=$(grep -rniE "$GT_PATTERNS" \
        "${PRISM_ROOT}/crates" \
        --include="*.rs" \
        --exclude-dir=test \
        2>/dev/null | head -10 || true)

    if [[ -n "$HARDCODED_RESIDUES" ]]; then
        fail "Found patterns matching ground truth residue lists:"
        echo "$HARDCODED_RESIDUES"
    else
        pass "No ground truth residue patterns found hardcoded"
    fi

    # Check for hardcoded druggability scores
    info "Checking for hardcoded druggability scores..."

    HARDCODED_SCORES=$(grep -rniE "druggability.*=.*0\.[5-9][0-9]{2}|0\.(580|726|889|792)" \
        "${PRISM_ROOT}/crates" \
        --include="*.rs" \
        --exclude-dir=test \
        2>/dev/null | head -10 || true)

    if [[ -n "$HARDCODED_SCORES" ]]; then
        warn "Found potential hardcoded druggability values:"
        echo "$HARDCODED_SCORES"
    else
        pass "No hardcoded druggability scores found"
    fi

    # Check for lookup tables by PDB ID
    info "Checking for PDB-keyed lookup tables..."

    LOOKUP_TABLES=$(grep -rniE "HashMap.*PDB|match.*pdb_id|pdb.*=>|\"[0-9][a-z0-9]{3}\".*=>" \
        "${PRISM_ROOT}/crates" \
        --include="*.rs" \
        --exclude-dir=test \
        2>/dev/null | head -10 || true)

    if [[ -n "$LOOKUP_TABLES" ]]; then
        warn "Found potential PDB-keyed lookup patterns (review manually):"
        echo "$LOOKUP_TABLES"
    else
        pass "No PDB-keyed lookup tables found"
    fi

    # Check for any file reads of ground truth
    info "Checking for ground truth file reads in production code..."

    GT_READS=$(grep -rniE "ground_truth|benchmark.*csv|tier[0-9].*csv|asbench|cryptosite" \
        "${PRISM_ROOT}/crates" \
        --include="*.rs" \
        --exclude="*test*" \
        --exclude="*bench*" \
        2>/dev/null | head -10 || true)

    if [[ -n "$GT_READS" ]]; then
        fail "Found ground truth references in production code:"
        echo "$GT_READS"
    else
        pass "No ground truth file references in production code"
    fi
}

# ============================================================================
# AUDIT 2: GPU MODULE VERIFICATION
# ============================================================================

audit_gpu_modules() {
    print_header "AUDIT 2: GPU MODULE VERIFICATION"

    log ""
    log "Verifying GPU modules exist and are properly integrated..."
    log ""

    # Expected GPU modules from world-class directive
    declare -a GPU_MODULES=(
        "gpu_floyd_warshall:allosteric:GPU Floyd-Warshall shortest paths"
        "gpu_lanczos:softspot:GPU Lanczos eigensolver"
    )

    declare -a CPU_MODULES=(
        "delaunay_detector:pocket:Delaunay alpha sphere detection"
        "lanczos:softspot:Lanczos eigensolver (CPU)"
        "sasa:pocket:Shrake-Rupley SASA"
        "hdbscan:pocket:HDBSCAN clustering"
    )

    # Check GPU modules
    info "Checking GPU modules..."

    for module_info in "${GPU_MODULES[@]}"; do
        IFS=':' read -r module_name subdir description <<< "$module_info"
        module_path="${PRISM_ROOT}/crates/prism-lbs/src/${subdir}/${module_name}.rs"

        if [[ -f "$module_path" ]]; then
            line_count=$(wc -l < "$module_path")

            # Check for actual CUDA/GPU code patterns
            gpu_patterns=$(grep -cE "cuda|gpu|kernel|device|wgpu|vulkan|compute|dispatch|workgroup" "$module_path" 2>/dev/null || echo "0")

            if [[ $line_count -lt 50 ]]; then
                fail "$description: Only $line_count lines (stub?)"
            elif [[ $gpu_patterns -lt 3 ]]; then
                warn "$description: $line_count lines but only $gpu_patterns GPU-related patterns"
            else
                pass "$description: $line_count lines, $gpu_patterns GPU patterns"
            fi

            # Check if exported in mod.rs
            mod_file="${PRISM_ROOT}/crates/prism-lbs/src/${subdir}/mod.rs"
            if [[ -f "$mod_file" ]]; then
                if grep -q "pub mod ${module_name}" "$mod_file" 2>/dev/null; then
                    pass "  └─ Exported in ${subdir}/mod.rs"
                else
                    fail "  └─ NOT exported in ${subdir}/mod.rs"
                fi
            fi
        else
            fail "$description: FILE NOT FOUND at $module_path"
        fi
    done

    # Check CPU modules
    info "Checking CPU modules..."

    for module_info in "${CPU_MODULES[@]}"; do
        IFS=':' read -r module_name subdir description <<< "$module_info"
        module_path="${PRISM_ROOT}/crates/prism-lbs/src/${subdir}/${module_name}.rs"

        if [[ -f "$module_path" ]]; then
            line_count=$(wc -l < "$module_path")

            if [[ $line_count -lt 50 ]]; then
                fail "$description: Only $line_count lines (stub?)"
            else
                pass "$description: $line_count lines"
            fi
        else
            warn "$description: File not found (may not be implemented)"
        fi
    done

    # Check for GPU feature flags in Cargo.toml
    info "Checking GPU feature flags in Cargo.toml..."

    cargo_toml="${PRISM_ROOT}/crates/prism-lbs/Cargo.toml"
    if [[ -f "$cargo_toml" ]]; then
        if grep -qE "cuda|gpu|wgpu|vulkan" "$cargo_toml" 2>/dev/null; then
            pass "GPU-related dependencies found in Cargo.toml"
            grep -E "cuda|gpu|wgpu|vulkan" "$cargo_toml" | head -5
        else
            warn "No obvious GPU dependencies in Cargo.toml"
        fi
    fi

    # Check if GPU code is actually called in main detection path
    info "Checking if GPU modules are called in detection pipeline..."

    lib_file="${PRISM_ROOT}/crates/prism-lbs/src/lib.rs"
    if [[ -f "$lib_file" ]]; then
        gpu_calls=$(grep -cE "gpu_floyd|gpu_lanczos|GpuFloyd|GpuLanczos" "$lib_file" 2>/dev/null || echo "0")

        if [[ $gpu_calls -gt 0 ]]; then
            pass "Found $gpu_calls GPU module references in lib.rs"
        else
            warn "No GPU module calls found in lib.rs main path"
        fi
    fi
}

# ============================================================================
# AUDIT 3: DETERMINISM AND REPRODUCIBILITY
# ============================================================================

audit_determinism() {
    print_header "AUDIT 3: DETERMINISM AND REPRODUCIBILITY"

    log ""
    log "Verifying results are deterministic and reproducible..."
    log ""

    PRISM_BINARY="${PRISM_ROOT}/target/release/prism-lbs"

    if [[ ! -f "$PRISM_BINARY" ]]; then
        warn "PRISM binary not found, skipping runtime tests"
        return
    fi

    # Download a test structure
    TEST_PDB="${AUDIT_DIR}/test_4hvp.pdb"
    if [[ ! -f "$TEST_PDB" ]]; then
        info "Downloading test structure 4HVP..."
        wget -q "https://files.rcsb.org/download/4HVP.pdb" -O "$TEST_PDB" 2>/dev/null || {
            warn "Could not download test structure"
            return
        }
    fi

    # Run PRISM multiple times and compare
    info "Running PRISM 3 times to verify determinism..."

    RUN1="${AUDIT_DIR}/run1.json"
    RUN2="${AUDIT_DIR}/run2.json"
    RUN3="${AUDIT_DIR}/run3.json"

    "$PRISM_BINARY" "$TEST_PDB" -o "$RUN1" 2>/dev/null
    "$PRISM_BINARY" "$TEST_PDB" -o "$RUN2" 2>/dev/null
    "$PRISM_BINARY" "$TEST_PDB" -o "$RUN3" 2>/dev/null

    # Extract key values
    get_fingerprint() {
        jq -r '[.pockets[] | {v: .volume, d: .druggability_score.total, r: .residue_indices | sort | join(",")}] | @json' "$1" 2>/dev/null
    }

    FP1=$(get_fingerprint "$RUN1")
    FP2=$(get_fingerprint "$RUN2")
    FP3=$(get_fingerprint "$RUN3")

    if [[ "$FP1" == "$FP2" ]] && [[ "$FP2" == "$FP3" ]]; then
        pass "Results are deterministic across 3 runs"
    else
        fail "Results differ between runs (non-deterministic)"
        log "Run 1: ${FP1:0:100}..."
        log "Run 2: ${FP2:0:100}..."
    fi

    # Verify pocket count is reasonable
    POCKET_COUNT=$(jq '.pockets | length' "$RUN1" 2>/dev/null)
    if [[ $POCKET_COUNT -gt 0 ]] && [[ $POCKET_COUNT -lt 100 ]]; then
        pass "Reasonable pocket count: $POCKET_COUNT"
    else
        warn "Unusual pocket count: $POCKET_COUNT"
    fi
}

# ============================================================================
# AUDIT 4: STRUCTURE-DEPENDENT OUTPUT
# ============================================================================

audit_structure_dependency() {
    print_header "AUDIT 4: STRUCTURE-DEPENDENT OUTPUT"

    log ""
    log "Verifying output changes with input structure..."
    log ""

    PRISM_BINARY="${PRISM_ROOT}/target/release/prism-lbs"

    if [[ ! -f "$PRISM_BINARY" ]]; then
        warn "PRISM binary not found, skipping runtime tests"
        return
    fi

    # Download two different structures
    STRUCT_A="${AUDIT_DIR}/test_1hpv.pdb"  # HIV protease
    STRUCT_B="${AUDIT_DIR}/test_1ubq.pdb"  # Ubiquitin (very different)

    if [[ ! -f "$STRUCT_A" ]]; then
        wget -q "https://files.rcsb.org/download/1HPV.pdb" -O "$STRUCT_A" 2>/dev/null
    fi
    if [[ ! -f "$STRUCT_B" ]]; then
        wget -q "https://files.rcsb.org/download/1UBQ.pdb" -O "$STRUCT_B" 2>/dev/null
    fi

    if [[ ! -f "$STRUCT_A" ]] || [[ ! -f "$STRUCT_B" ]]; then
        warn "Could not download test structures"
        return
    fi

    info "Running PRISM on two different structures..."

    OUT_A="${AUDIT_DIR}/output_1hpv.json"
    OUT_B="${AUDIT_DIR}/output_1ubq.json"

    "$PRISM_BINARY" "$STRUCT_A" -o "$OUT_A" 2>/dev/null
    "$PRISM_BINARY" "$STRUCT_B" -o "$OUT_B" 2>/dev/null

    # Compare outputs
    POCKETS_A=$(jq '.pockets | length' "$OUT_A" 2>/dev/null)
    POCKETS_B=$(jq '.pockets | length' "$OUT_B" 2>/dev/null)

    VOL_A=$(jq '.pockets[0].volume // 0' "$OUT_A" 2>/dev/null)
    VOL_B=$(jq '.pockets[0].volume // 0' "$OUT_B" 2>/dev/null)

    DRUG_A=$(jq '.pockets[0].druggability_score.total // 0' "$OUT_A" 2>/dev/null)
    DRUG_B=$(jq '.pockets[0].druggability_score.total // 0' "$OUT_B" 2>/dev/null)

    RES_A=$(jq '.pockets[0].residue_indices | sort | join(",")' "$OUT_A" 2>/dev/null)
    RES_B=$(jq '.pockets[0].residue_indices | sort | join(",")' "$OUT_B" 2>/dev/null)

    info "1HPV (HIV protease): $POCKETS_A pockets, vol=$VOL_A, drug=$DRUG_A"
    info "1UBQ (Ubiquitin):    $POCKETS_B pockets, vol=$VOL_B, drug=$DRUG_B"

    # Check that outputs are different
    if [[ "$RES_A" != "$RES_B" ]]; then
        pass "Different structures produce different residue sets"
    else
        fail "Same residues detected for completely different proteins!"
    fi

    if [[ "$VOL_A" != "$VOL_B" ]]; then
        pass "Different structures produce different volumes"
    else
        warn "Same volume for different proteins (suspicious)"
    fi

    if [[ "$DRUG_A" != "$DRUG_B" ]]; then
        pass "Different structures produce different druggability scores"
    else
        warn "Same druggability for different proteins (suspicious)"
    fi
}

# ============================================================================
# AUDIT 5: ALGORITHM VERIFICATION
# ============================================================================

audit_algorithms() {
    print_header "AUDIT 5: ALGORITHM IMPLEMENTATION VERIFICATION"

    log ""
    log "Verifying core algorithms are implemented (not stubbed)..."
    log ""

    # Check alpha sphere detection
    info "Checking alpha sphere / Voronoi detection..."

    VORONOI_FILE="${PRISM_ROOT}/crates/prism-lbs/src/pocket/voronoi_detector.rs"
    if [[ -f "$VORONOI_FILE" ]]; then
        voronoi_lines=$(wc -l < "$VORONOI_FILE")

        # Check for key algorithmic patterns
        has_delaunay=$(grep -c "delaunay\|tetrahedr\|circumsphere" "$VORONOI_FILE" 2>/dev/null || echo "0")
        has_alpha=$(grep -c "alpha.*sphere\|alpha_sphere\|probe.*radius" "$VORONOI_FILE" 2>/dev/null || echo "0")
        has_clustering=$(grep -c "cluster\|dbscan\|hdbscan\|group" "$VORONOI_FILE" 2>/dev/null || echo "0")

        if [[ $voronoi_lines -gt 200 ]] && [[ $has_alpha -gt 2 ]]; then
            pass "Voronoi detector: $voronoi_lines lines, $has_alpha alpha sphere refs"
        else
            warn "Voronoi detector may be incomplete: $voronoi_lines lines, $has_alpha alpha refs"
        fi
    else
        fail "Voronoi detector not found"
    fi

    # Check druggability scoring
    info "Checking druggability scoring implementation..."

    DRUG_FILE="${PRISM_ROOT}/crates/prism-lbs/src/druggability.rs"
    if [[ -f "$DRUG_FILE" ]]; then
        drug_lines=$(wc -l < "$DRUG_FILE")

        # Check for druggability factors
        has_hydrophob=$(grep -c "hydrophob\|lipophil" "$DRUG_FILE" 2>/dev/null || echo "0")
        has_enclosure=$(grep -c "enclosure\|burial\|enclosed" "$DRUG_FILE" 2>/dev/null || echo "0")
        has_volume=$(grep -c "volume\|size" "$DRUG_FILE" 2>/dev/null || echo "0")

        if [[ $drug_lines -gt 100 ]] && [[ $has_hydrophob -gt 0 ]] && [[ $has_enclosure -gt 0 ]]; then
            pass "Druggability scoring: $drug_lines lines, hydrophob=$has_hydrophob, enclosure=$has_enclosure"
        else
            warn "Druggability scoring may be simplified"
        fi
    else
        # Check in unified.rs or lib.rs
        LIB_FILE="${PRISM_ROOT}/crates/prism-lbs/src/lib.rs"
        if [[ -f "$LIB_FILE" ]]; then
            drug_refs=$(grep -c "druggability\|Druggability" "$LIB_FILE" 2>/dev/null || echo "0")
            if [[ $drug_refs -gt 5 ]]; then
                pass "Druggability scoring found in lib.rs ($drug_refs references)"
            else
                warn "Limited druggability scoring references"
            fi
        fi
    fi

    # Check flexibility/NMA
    info "Checking flexibility analysis..."

    FLEX_FILE="${PRISM_ROOT}/crates/prism-lbs/src/softspot/flexibility.rs"
    LANCZOS_FILE="${PRISM_ROOT}/crates/prism-lbs/src/softspot/lanczos.rs"

    if [[ -f "$FLEX_FILE" ]] || [[ -f "$LANCZOS_FILE" ]]; then
        flex_lines=0
        [[ -f "$FLEX_FILE" ]] && flex_lines=$(wc -l < "$FLEX_FILE")
        lanczos_lines=0
        [[ -f "$LANCZOS_FILE" ]] && lanczos_lines=$(wc -l < "$LANCZOS_FILE")

        total_flex=$((flex_lines + lanczos_lines))

        if [[ $total_flex -gt 200 ]]; then
            pass "Flexibility analysis: $total_flex total lines"
        else
            warn "Flexibility analysis may be limited: $total_flex lines"
        fi
    else
        warn "No dedicated flexibility module found"
    fi

    # Check pocket merging (key to your 100% result)
    info "Checking pocket merging implementation..."

    LIB_FILE="${PRISM_ROOT}/crates/prism-lbs/src/lib.rs"
    if [[ -f "$LIB_FILE" ]]; then
        merge_refs=$(grep -c "merge.*pocket\|pocket.*merg\|adjacent.*pocket" "$LIB_FILE" 2>/dev/null || echo "0")
        merge_func=$(grep -c "fn.*merge" "$LIB_FILE" 2>/dev/null || echo "0")

        if [[ $merge_refs -gt 3 ]] && [[ $merge_func -gt 0 ]]; then
            pass "Pocket merging: $merge_refs refs, $merge_func functions"
        else
            warn "Limited pocket merging implementation found"
        fi
    fi
}

# ============================================================================
# AUDIT 6: TIMING ANALYSIS (ANTI-CHEAT)
# ============================================================================

audit_timing() {
    print_header "AUDIT 6: TIMING ANALYSIS (ANTI-CHEAT)"

    log ""
    log "Verifying runtime scales with input size..."
    log ""

    PRISM_BINARY="${PRISM_ROOT}/target/release/prism-lbs"

    if [[ ! -f "$PRISM_BINARY" ]]; then
        warn "PRISM binary not found, skipping timing tests"
        return
    fi

    # Test with small and large structures
    SMALL_PDB="${AUDIT_DIR}/test_1ubq.pdb"   # ~76 residues
    LARGE_PDB="${AUDIT_DIR}/test_4hhb.pdb"   # ~574 residues (hemoglobin)

    if [[ ! -f "$SMALL_PDB" ]]; then
        wget -q "https://files.rcsb.org/download/1UBQ.pdb" -O "$SMALL_PDB" 2>/dev/null
    fi
    if [[ ! -f "$LARGE_PDB" ]]; then
        wget -q "https://files.rcsb.org/download/4HHB.pdb" -O "$LARGE_PDB" 2>/dev/null
    fi

    if [[ ! -f "$SMALL_PDB" ]] || [[ ! -f "$LARGE_PDB" ]]; then
        warn "Could not download test structures for timing"
        return
    fi

    SMALL_ATOMS=$(grep -c "^ATOM" "$SMALL_PDB" 2>/dev/null || echo "0")
    LARGE_ATOMS=$(grep -c "^ATOM" "$LARGE_PDB" 2>/dev/null || echo "0")

    info "Small structure: $SMALL_ATOMS atoms"
    info "Large structure: $LARGE_ATOMS atoms"

    # Time small structure
    START=$(date +%s.%N)
    "$PRISM_BINARY" "$SMALL_PDB" -o "${AUDIT_DIR}/timing_small.json" 2>/dev/null
    END=$(date +%s.%N)
    TIME_SMALL=$(echo "$END - $START" | bc)

    # Time large structure
    START=$(date +%s.%N)
    "$PRISM_BINARY" "$LARGE_PDB" -o "${AUDIT_DIR}/timing_large.json" 2>/dev/null
    END=$(date +%s.%N)
    TIME_LARGE=$(echo "$END - $START" | bc)

    info "Small structure time: ${TIME_SMALL}s"
    info "Large structure time: ${TIME_LARGE}s"

    # Large should take longer (if real computation is happening)
    RATIO=$(echo "$TIME_LARGE / $TIME_SMALL" | bc -l 2>/dev/null || echo "1")

    if (( $(echo "$TIME_LARGE > $TIME_SMALL" | bc -l) )); then
        pass "Larger structure takes longer (${RATIO}x ratio) - computation is real"
    else
        fail "Large structure NOT slower than small - suspicious!"
    fi

    # Check if times are suspiciously fast (instant lookup)
    if (( $(echo "$TIME_SMALL < 0.01" | bc -l) )); then
        warn "Small structure completed in <10ms - suspiciously fast"
    fi
}

# ============================================================================
# AUDIT 7: BINARY INSPECTION
# ============================================================================

audit_binary() {
    print_header "AUDIT 7: BINARY INSPECTION"

    log ""
    log "Inspecting compiled binary for suspicious content..."
    log ""

    PRISM_BINARY="${PRISM_ROOT}/target/release/prism-lbs"

    if [[ ! -f "$PRISM_BINARY" ]]; then
        warn "PRISM binary not found"
        return
    fi

    # Check binary size
    BINARY_SIZE=$(stat -c%s "$PRISM_BINARY" 2>/dev/null || stat -f%z "$PRISM_BINARY" 2>/dev/null)
    BINARY_SIZE_MB=$(echo "scale=1; $BINARY_SIZE / 1048576" | bc)

    info "Binary size: ${BINARY_SIZE_MB}MB"

    if (( $(echo "$BINARY_SIZE_MB < 1" | bc -l) )); then
        warn "Binary is very small (<1MB) - may be missing features"
    elif (( $(echo "$BINARY_SIZE_MB > 100" | bc -l) )); then
        warn "Binary is very large (>100MB) - may contain embedded data"
    else
        pass "Binary size is reasonable (${BINARY_SIZE_MB}MB)"
    fi

    # Check for embedded PDB data
    info "Checking for embedded PDB/benchmark data in binary..."

    EMBEDDED_PDBS=$(strings "$PRISM_BINARY" 2>/dev/null | grep -ciE "^(ATOM|HETATM|TER|END)" || echo "0")

    if [[ $EMBEDDED_PDBS -gt 100 ]]; then
        fail "Found $EMBEDDED_PDBS PDB-like strings embedded in binary!"
    else
        pass "No significant embedded PDB data found"
    fi

    # Check for ground truth data
    GT_STRINGS=$(strings "$PRISM_BINARY" 2>/dev/null | grep -ciE "ground.truth|benchmark|tier[0-9]" || echo "0")

    if [[ $GT_STRINGS -gt 10 ]]; then
        warn "Found $GT_STRINGS ground truth references in binary"
    else
        pass "Minimal ground truth references in binary"
    fi
}

# ============================================================================
# AUDIT 8: SOURCE CODE QUALITY
# ============================================================================

audit_code_quality() {
    print_header "AUDIT 8: SOURCE CODE QUALITY"

    log ""
    log "Checking source code patterns..."
    log ""

    SRC_DIR="${PRISM_ROOT}/crates/prism-lbs/src"

    if [[ ! -d "$SRC_DIR" ]]; then
        fail "Source directory not found"
        return
    fi

    # Count total lines of code
    TOTAL_LINES=$(find "$SRC_DIR" -name "*.rs" -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}')
    info "Total Rust source lines: $TOTAL_LINES"

    # Check for TODO/FIXME/HACK comments
    TODO_COUNT=$(grep -rci "TODO\|FIXME\|HACK\|XXX" "$SRC_DIR" --include="*.rs" 2>/dev/null | awk -F: '{sum+=$2} END {print sum}' || echo "0")
    if [[ $TODO_COUNT -gt 50 ]]; then
        warn "$TODO_COUNT TODO/FIXME comments found"
    else
        info "$TODO_COUNT TODO/FIXME comments"
    fi

    # Check for panic/unwrap usage
    UNWRAP_COUNT=$(grep -rc "\.unwrap()\|panic!" "$SRC_DIR" --include="*.rs" 2>/dev/null | awk -F: '{sum+=$2} END {print sum}' || echo "0")
    if [[ $UNWRAP_COUNT -gt 100 ]]; then
        warn "$UNWRAP_COUNT unwrap()/panic! calls - could cause crashes"
    else
        info "$UNWRAP_COUNT unwrap()/panic! calls"
    fi

    # Check for tests
    TEST_COUNT=$(grep -rc "#\[test\]" "$SRC_DIR" --include="*.rs" 2>/dev/null | awk -F: '{sum+=$2} END {print sum}' || echo "0")
    if [[ $TEST_COUNT -lt 10 ]]; then
        warn "Only $TEST_COUNT unit tests found"
    else
        pass "$TEST_COUNT unit tests found"
    fi

    # Check for doc comments
    DOC_COUNT=$(grep -rc "///" "$SRC_DIR" --include="*.rs" 2>/dev/null | awk -F: '{sum+=$2} END {print sum}' || echo "0")
    if [[ $DOC_COUNT -lt 50 ]]; then
        warn "Only $DOC_COUNT doc comments found"
    else
        pass "$DOC_COUNT doc comments found"
    fi
}

# ============================================================================
# AUDIT 9: VALIDATION SCRIPT INTEGRITY
# ============================================================================

audit_validation_scripts() {
    print_header "AUDIT 9: VALIDATION SCRIPT INTEGRITY"

    log ""
    log "Checking validation scripts for manipulation..."
    log ""

    SCRIPTS_DIR="${PRISM_ROOT}/scripts"

    if [[ ! -d "$SCRIPTS_DIR" ]]; then
        warn "Scripts directory not found"
        return
    fi

    # Check for hardcoded pass conditions
    info "Checking for hardcoded pass conditions in validation scripts..."

    HARDCODED_PASS=$(grep -rniE "overlap.*=.*[0-9]{2}\.[0-9]|PASS.*hardcode|force.*pass|always.*pass" \
        "$SCRIPTS_DIR" \
        --include="*.sh" \
        2>/dev/null | head -10 || true)

    if [[ -n "$HARDCODED_PASS" ]]; then
        fail "Found potential hardcoded pass conditions:"
        echo "$HARDCODED_PASS"
    else
        pass "No hardcoded pass conditions in validation scripts"
    fi

    # Check that validation actually reads PRISM output
    info "Checking that validation reads actual PRISM output..."

    VAL_SCRIPT="${SCRIPTS_DIR}/prism_validation_suite_v2.1_aligned.sh"
    if [[ -f "$VAL_SCRIPT" ]]; then
        reads_json=$(grep -c "jq\|\.json" "$VAL_SCRIPT" 2>/dev/null || echo "0")
        runs_prism=$(grep -c "prism-lbs\|PRISM_BINARY" "$VAL_SCRIPT" 2>/dev/null || echo "0")

        if [[ $reads_json -gt 5 ]] && [[ $runs_prism -gt 0 ]]; then
            pass "Validation script properly runs PRISM and parses JSON output"
        else
            warn "Validation script may not properly run PRISM"
        fi
    else
        warn "Main validation script not found"
    fi
}

# ============================================================================
# AUDIT 10: MODULE CALL TRACE
# ============================================================================

audit_call_trace() {
    print_header "AUDIT 10: MODULE CALL TRACE ANALYSIS"

    log ""
    log "Tracing which modules are actually called in the detection pipeline..."
    log ""

    LIB_FILE="${PRISM_ROOT}/crates/prism-lbs/src/lib.rs"
    UNIFIED_FILE="${PRISM_ROOT}/crates/prism-lbs/src/unified.rs"

    if [[ ! -f "$LIB_FILE" ]]; then
        fail "lib.rs not found"
        return
    fi

    info "Analyzing main detection path in lib.rs..."

    # Look for the main detection function
    MAIN_FUNC=$(grep -n "pub fn detect\|pub fn analyze\|pub fn run" "$LIB_FILE" 2>/dev/null | head -5)
    if [[ -n "$MAIN_FUNC" ]]; then
        log "Main detection functions found:"
        echo "$MAIN_FUNC"
    fi

    # Check what modules are imported and used
    info "Checking module imports and usage..."

    declare -A MODULE_STATUS

    # List of expected modules
    MODULES=("voronoi" "cavity" "delaunay" "druggability" "flexibility" "lanczos" "hdbscan" "sasa" "floyd" "consensus")

    for module in "${MODULES[@]}"; do
        import_count=$(grep -ci "use.*${module}\|mod ${module}\|${module}::" "$LIB_FILE" 2>/dev/null || echo "0")
        call_count=$(grep -ci "${module}" "$LIB_FILE" 2>/dev/null || echo "0")

        if [[ $call_count -gt 2 ]]; then
            pass "$module: $call_count references in lib.rs"
        elif [[ $call_count -gt 0 ]]; then
            warn "$module: only $call_count references (may be unused)"
        else
            warn "$module: NOT FOUND in lib.rs"
        fi
    done

    # Check for dead code
    info "Checking for potentially dead modules..."

    if [[ -f "$UNIFIED_FILE" ]]; then
        unified_lines=$(wc -l < "$UNIFIED_FILE")
        info "unified.rs: $unified_lines lines"
    fi
}

# ============================================================================
# SUMMARY
# ============================================================================

print_summary() {
    print_header "AUDIT SUMMARY"

    log ""
    log "═══════════════════════════════════════════════════════════════════════════════"

    if [[ $FAILURES -eq 0 ]] && [[ $WARNINGS -lt 5 ]]; then
        log "${GREEN}${BOLD}  ✅ AUDIT PASSED${NC}"
        log ""
        log "  The PRISM-LBS implementation appears legitimate."
        log "  No evidence of hardcoding, data leakage, or shortcuts found."
    elif [[ $FAILURES -eq 0 ]]; then
        log "${YELLOW}${BOLD}  ⚠️  AUDIT PASSED WITH WARNINGS${NC}"
        log ""
        log "  No critical issues found, but $WARNINGS warnings need review."
    else
        log "${RED}${BOLD}  ❌ AUDIT FAILED${NC}"
        log ""
        log "  Found $FAILURES critical issues that require investigation."
    fi

    log ""
    log "  Failures: $FAILURES"
    log "  Warnings: $WARNINGS"
    log ""
    log "  Full log: $AUDIT_LOG"
    log "═══════════════════════════════════════════════════════════════════════════════"
}

# ============================================================================
# MAIN
# ============================================================================

main() {
    print_header "PRISM-LBS COMPREHENSIVE INTEGRITY AUDIT"

    log ""
    log "Timestamp: $(date)"
    log "PRISM Root: $PRISM_ROOT"
    log ""

    audit_hardcoded_values
    audit_gpu_modules
    audit_algorithms
    audit_determinism
    audit_structure_dependency
    audit_timing
    audit_binary
    audit_code_quality
    audit_validation_scripts
    audit_call_trace

    print_summary
}

main "$@"
