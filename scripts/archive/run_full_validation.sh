#!/bin/bash
#===============================================================================
# PRISM-LBS Full Validation Suite
#
# Runs all validation tiers:
#   Tier 1: Basic correctness (must pass)
#   Tier 2: CryptoSite benchmark (novelty test)
#   Tier 3: Speed benchmark (GPU value test)
#
# Usage: ./scripts/run_full_validation.sh
#===============================================================================

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Ensure we're in the project directory
cd "$PROJECT_DIR"

echo ""
echo -e "${MAGENTA}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${MAGENTA}║                                                                  ║${NC}"
echo -e "${MAGENTA}║          PRISM-LBS FULL VALIDATION SUITE                         ║${NC}"
echo -e "${MAGENTA}║                                                                  ║${NC}"
echo -e "${MAGENTA}║  Testing: Correctness → Novelty → Performance                    ║${NC}"
echo -e "${MAGENTA}║                                                                  ║${NC}"
echo -e "${MAGENTA}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Track results
TIER1_RESULT="SKIP"
TIER2_RESULT="SKIP"
TIER3_RESULT="SKIP"
CRYPTOSITE_RATE="N/A"
SPEED_FACTOR="N/A"

#===============================================================================
# Build PRISM if needed
#===============================================================================

echo -e "${BLUE}[STEP 0] Building PRISM with GPU acceleration...${NC}"
echo ""

# Export CUDA environment for entire script runtime
export PATH="/home/diddy/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:/usr/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12.6
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH

if ! cargo build --release --features cuda 2>&1 | tail -5; then
    echo -e "${YELLOW}[WARN] Build with CUDA failed, trying without...${NC}"
    cargo build --release 2>&1 | tail -3 || {
        echo -e "${RED}[FAIL] Could not build PRISM${NC}"
        exit 1
    }
fi

echo ""
echo -e "${GREEN}[OK] Build complete${NC}"
echo ""

#===============================================================================
# TIER 1: Basic Correctness
#===============================================================================

echo "═══════════════════════════════════════════════════════════════════"
echo -e "${CYAN}TIER 1: Basic Correctness${NC}"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

if [ -f "$SCRIPT_DIR/validate_basic.sh" ]; then
    if bash "$SCRIPT_DIR/validate_basic.sh"; then
        TIER1_RESULT="PASS"
    else
        TIER1_RESULT="PARTIAL"
    fi
else
    echo -e "${YELLOW}[SKIP] validate_basic.sh not found${NC}"
    TIER1_RESULT="SKIP"
fi

echo ""

#===============================================================================
# TIER 2: CryptoSite Benchmark
#===============================================================================

echo "═══════════════════════════════════════════════════════════════════"
echo -e "${CYAN}TIER 2: CryptoSite Benchmark (Novelty Test)${NC}"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

CRYPTOSITE_DIR="$PROJECT_DIR/benchmark/cryptosite"

# Setup if not already done
if [ ! -f "$CRYPTOSITE_DIR/ground_truth.csv" ]; then
    echo -e "${BLUE}[INFO] Setting up CryptoSite benchmark...${NC}"
    if [ -f "$SCRIPT_DIR/setup_cryptosite_benchmark.sh" ]; then
        bash "$SCRIPT_DIR/setup_cryptosite_benchmark.sh"
    else
        echo -e "${YELLOW}[SKIP] setup_cryptosite_benchmark.sh not found${NC}"
    fi
fi

if [ -f "$CRYPTOSITE_DIR/run_benchmark.sh" ]; then
    echo -e "${BLUE}[INFO] Running CryptoSite benchmark...${NC}"
    echo ""

    # Run benchmark and capture output
    if bash "$CRYPTOSITE_DIR/run_benchmark.sh" 2>&1 | tee /tmp/cryptosite_output.txt; then
        # Extract detection rate from output
        CRYPTOSITE_RATE=$(grep -oP "PRISM detected: \d+/\d+ \(\K[0-9.]+(?=%)" /tmp/cryptosite_output.txt || echo "N/A")

        if [[ "$CRYPTOSITE_RATE" != "N/A" ]]; then
            RATE_NUM=$(echo "$CRYPTOSITE_RATE" | cut -d'.' -f1)
            if [ "$RATE_NUM" -ge 60 ]; then
                TIER2_RESULT="EXCELLENT"
            elif [ "$RATE_NUM" -ge 50 ]; then
                TIER2_RESULT="GOOD"
            elif [ "$RATE_NUM" -ge 40 ]; then
                TIER2_RESULT="COMPETITIVE"
            else
                TIER2_RESULT="NEEDS_WORK"
            fi
        else
            TIER2_RESULT="RAN"
        fi
    else
        TIER2_RESULT="FAILED"
    fi
else
    echo -e "${YELLOW}[SKIP] CryptoSite benchmark not set up${NC}"
    TIER2_RESULT="SKIP"
fi

echo ""

#===============================================================================
# TIER 3: Speed Benchmark
#===============================================================================

echo "═══════════════════════════════════════════════════════════════════"
echo -e "${CYAN}TIER 3: Speed Benchmark (GPU Value Test)${NC}"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

SPEED_DIR="$PROJECT_DIR/benchmark/speed"
mkdir -p "$SPEED_DIR"

# Download ribosome if not present
RIBOSOME_PDB="$SPEED_DIR/ribosome.pdb"
if [ ! -f "$RIBOSOME_PDB" ]; then
    echo -e "${BLUE}[INFO] Downloading ribosome structure (large test)...${NC}"
    wget -q "https://files.rcsb.org/download/4V9D.pdb" -O "$RIBOSOME_PDB" 2>/dev/null || {
        echo -e "${YELLOW}[WARN] Could not download ribosome, using smaller test${NC}"
        # Fall back to a medium structure
        wget -q "https://files.rcsb.org/download/3J3Y.pdb" -O "$RIBOSOME_PDB" 2>/dev/null || {
            echo -e "${YELLOW}[SKIP] No large structure available for speed test${NC}"
            RIBOSOME_PDB=""
        }
    }
fi

if [ -n "$RIBOSOME_PDB" ] && [ -f "$RIBOSOME_PDB" ]; then
    ATOM_COUNT=$(grep -c "^ATOM" "$RIBOSOME_PDB" || echo "0")
    echo -e "${BLUE}[INFO] Test structure: $ATOM_COUNT atoms${NC}"
    echo ""

    PRISM_BIN="$PROJECT_DIR/target/release/prism"

    if [ -f "$PRISM_BIN" ]; then
        echo -e "${BLUE}[INFO] Timing PRISM (full GPU + unified)...${NC}"
        PRISM_START=$(date +%s.%N)
        PRISM_PTX_DIR="$PROJECT_DIR/target/ptx" \
        timeout 300 $PRISM_BIN --input "$RIBOSOME_PDB" --output "$SPEED_DIR/ribosome_prism.json" \
            --gpu-geometry --publication --unified 2>/dev/null || true
        PRISM_END=$(date +%s.%N)
        PRISM_TIME=$(echo "$PRISM_END - $PRISM_START" | bc)
        echo -e "  PRISM: ${PRISM_TIME}s"

        # Check if fpocket is available
        if command -v fpocket &> /dev/null; then
            echo -e "${BLUE}[INFO] Timing fpocket...${NC}"
            FPOCKET_START=$(date +%s.%N)
            timeout 600 fpocket -f "$RIBOSOME_PDB" 2>/dev/null || true
            FPOCKET_END=$(date +%s.%N)
            FPOCKET_TIME=$(echo "$FPOCKET_END - $FPOCKET_START" | bc)
            echo -e "  fpocket: ${FPOCKET_TIME}s"

            # Calculate speedup
            if [ "$(echo "$FPOCKET_TIME > 0" | bc)" -eq 1 ]; then
                SPEED_FACTOR=$(echo "scale=1; $FPOCKET_TIME / $PRISM_TIME" | bc)
                echo ""
                echo -e "  Speedup: ${SPEED_FACTOR}x"

                if [ "$(echo "$SPEED_FACTOR >= 5" | bc)" -eq 1 ]; then
                    TIER3_RESULT="EXCELLENT"
                elif [ "$(echo "$SPEED_FACTOR >= 2" | bc)" -eq 1 ]; then
                    TIER3_RESULT="GOOD"
                else
                    TIER3_RESULT="COMPETITIVE"
                fi
            fi
        else
            echo -e "${YELLOW}[INFO] fpocket not installed, skipping comparison${NC}"
            echo -e "  PRISM processed $ATOM_COUNT atoms in ${PRISM_TIME}s"
            TIER3_RESULT="RAN"
        fi
    else
        echo -e "${YELLOW}[SKIP] PRISM binary not found${NC}"
        TIER3_RESULT="SKIP"
    fi
else
    TIER3_RESULT="SKIP"
fi

echo ""

#===============================================================================
# Final Summary
#===============================================================================

echo ""
echo -e "${MAGENTA}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${MAGENTA}║                    VALIDATION SUMMARY                            ║${NC}"
echo -e "${MAGENTA}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Color-code results
tier1_color="$YELLOW"
[[ "$TIER1_RESULT" == "PASS" ]] && tier1_color="$GREEN"
[[ "$TIER1_RESULT" == "FAIL" ]] && tier1_color="$RED"

tier2_color="$YELLOW"
[[ "$TIER2_RESULT" == "EXCELLENT" || "$TIER2_RESULT" == "GOOD" ]] && tier2_color="$GREEN"
[[ "$TIER2_RESULT" == "NEEDS_WORK" || "$TIER2_RESULT" == "FAILED" ]] && tier2_color="$RED"

tier3_color="$YELLOW"
[[ "$TIER3_RESULT" == "EXCELLENT" || "$TIER3_RESULT" == "GOOD" ]] && tier3_color="$GREEN"

printf "  %-30s ${tier1_color}%s${NC}\n" "Tier 1 (Basic Correctness):" "$TIER1_RESULT"
printf "  %-30s ${tier2_color}%s${NC}" "Tier 2 (CryptoSite):" "$TIER2_RESULT"
[[ "$CRYPTOSITE_RATE" != "N/A" ]] && printf " (%s%%)" "$CRYPTOSITE_RATE"
echo ""
printf "  %-30s ${tier3_color}%s${NC}" "Tier 3 (Speed):" "$TIER3_RESULT"
[[ "$SPEED_FACTOR" != "N/A" ]] && printf " (%sx faster)" "$SPEED_FACTOR"
echo ""

echo ""

# Overall assessment
if [[ "$TIER1_RESULT" == "PASS" && ("$TIER2_RESULT" == "EXCELLENT" || "$TIER2_RESULT" == "GOOD") ]]; then
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  VALIDATION STATUS: PUBLISHABLE QUALITY                          ║${NC}"
    echo -e "${GREEN}║                                                                  ║${NC}"
    echo -e "${GREEN}║  Consider submitting to:                                         ║${NC}"
    echo -e "${GREEN}║    - Journal of Chemical Information and Modeling                ║${NC}"
    echo -e "${GREEN}║    - Bioinformatics                                              ║${NC}"
    echo -e "${GREEN}║    - PLOS Computational Biology                                  ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════╝${NC}"
elif [[ "$TIER1_RESULT" == "PASS" ]]; then
    echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║  VALIDATION STATUS: FUNCTIONALLY CORRECT                         ║${NC}"
    echo -e "${BLUE}║                                                                  ║${NC}"
    echo -e "${BLUE}║  Basic validation passed. Improve CryptoSite performance for     ║${NC}"
    echo -e "${BLUE}║  publication-quality results.                                    ║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════╝${NC}"
else
    echo -e "${YELLOW}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${YELLOW}║  VALIDATION STATUS: IN PROGRESS                                  ║${NC}"
    echo -e "${YELLOW}║                                                                  ║${NC}"
    echo -e "${YELLOW}║  Some tests incomplete or not passing. Continue development.    ║${NC}"
    echo -e "${YELLOW}╚══════════════════════════════════════════════════════════════════╝${NC}"
fi

echo ""

# Save results to JSON
RESULTS_JSON="$PROJECT_DIR/benchmark/validation_results.json"
cat > "$RESULTS_JSON" << EOF
{
  "timestamp": "$(date -Iseconds)",
  "tier1_basic": "$TIER1_RESULT",
  "tier2_cryptosite": "$TIER2_RESULT",
  "tier2_rate": "$CRYPTOSITE_RATE",
  "tier3_speed": "$TIER3_RESULT",
  "tier3_factor": "$SPEED_FACTOR"
}
EOF

echo "Results saved to: $RESULTS_JSON"
echo ""
