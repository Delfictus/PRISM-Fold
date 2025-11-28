# PRISM Hypertuning Campaign - Quick Start Guide

## Overview

The PRISM hypertuning campaign system automatically explores the parameter space to achieve the world record goal: **17 colors with 0 conflicts** on DSJC500.5.

Based on recent telemetry showing:
- Phase 2 achieves **13 colors with 103 conflicts** (μ=0.6)
- Phase 3 achieves **17 colors with 58 conflicts** ⭐ (EXACTLY our target!)
- Conflict repair increases to 22-23 colors

The campaign focuses on maintaining the 17-color quantum solution while eliminating conflicts through intelligent parameter tuning.

## Three Core Scripts

### 1. world_record_campaign.sh
**Main orchestration script** - Runs automated parameter sweep with smart exploration.

**Features:**
- Intelligent 3-stage parameter selection (grid → refinement → aggressive)
- Automatic GPU kernel recompilation for chemical potential changes
- Resume capability via state tracking
- Early stopping when target achieved
- Comprehensive result tracking

### 2. monitor_campaign.sh
**Real-time monitoring dashboard** - Live view of campaign progress.

**Features:**
- Color-coded iteration results
- Progress bar with completion percentage
- Trend sparklines for chromatic/conflicts
- Best result tracking
- Current activity status
- Auto-refresh every 5 seconds

### 3. analyze_campaign_results.sh
**Post-campaign analysis** - Deep dive into results with recommendations.

**Features:**
- Statistical analysis of all iterations
- Parameter correlation analysis
- Phase-by-phase performance breakdown
- Conflict repair effectiveness evaluation
- Automated recommendations
- CSV export and Python visualization

## Quick Start

### Running a Campaign

```bash
# Basic run (uses default config and DSJC500.5)
cd /mnt/c/Users/Predator/Desktop/PRISM
./scripts/world_record_campaign.sh

# Custom campaign with specific config
./scripts/world_record_campaign.sh my_campaign configs/WORLD_RECORD_ATTEMPT.toml

# Custom graph
./scripts/world_record_campaign.sh test_run configs/WORLD_RECORD_ATTEMPT.toml data/graphs/DSJC125.5.col
```

The campaign will:
1. Create workspace at `campaigns/<campaign_name>/`
2. Run 20 iterations with parameter exploration
3. Stop early if 17 colors + 0 conflicts achieved
4. Generate summary report automatically

### Monitoring Live Progress

**In a separate terminal:**
```bash
# Auto-detect latest campaign
./scripts/monitor_campaign.sh

# Monitor specific campaign
./scripts/monitor_campaign.sh campaigns/world_record_20250123_140530

# Custom refresh rate (seconds)
./scripts/monitor_campaign.sh campaigns/world_record_20250123_140530 2
```

**What you'll see:**
- Real-time iteration results
- Color-coded status (green = perfect, yellow = suboptimal, red = failed)
- Progress bar toward target
- Trend sparklines
- Current best result

### Analyzing Results

**After campaign completes:**
```bash
# Analyze latest campaign
./scripts/analyze_campaign_results.sh

# Analyze specific campaign
./scripts/analyze_campaign_results.sh campaigns/world_record_20250123_140530
```

**Outputs:**
- `analysis/campaign_results.csv` - All iteration metrics
- `analysis/statistical_report.txt` - Statistical summary
- `analysis/phase_analysis.txt` - Phase-by-phase breakdown
- `analysis/recommendations.txt` - Next steps and optimal parameters
- `analysis/plot_results.py` - Visualization script

**Generate plots:**
```bash
# Requires: pip install matplotlib seaborn pandas
python3 campaigns/world_record_20250123_140530/analysis/plot_results.py \
        campaigns/world_record_20250123_140530/analysis/campaign_results.csv
```

## Parameter Exploration Strategy

### Stage 1: Grid Search (Iterations 1-8)
Systematic exploration of parameter space:
- Chemical potential (μ): 0.50, 0.55, 0.60, 0.65, 0.70
- Quantum coupling: 7.0, 7.5, 8.0, 8.5, 9.0
- Evolution time: Fixed at 0.14
- Conflict repair: Fixed at 1

### Stage 2: Refinement (Iterations 9-14)
Explores around best parameters found:
- Perturbs best μ by ±0.02
- Perturbs best coupling by ±0.5
- Perturbs evolution time by ±0.02
- Varies conflict repair increase: 0-2

### Stage 3: Aggressive Exploration (Iterations 15+)
Tries extreme combinations if target not achieved:
- Low μ (0.50) + high coupling (9.0) + long evolution (0.18)
- High μ (0.70) + low coupling (7.0) + short evolution (0.12)
- Balanced middle values
- Random conflict repair strategies

## Key Metrics Tracked

### Primary Objectives
- **Chromatic Number**: Target ≤ 17
- **Conflicts**: Target = 0
- **Combined Score**: chromatic × 100 + conflicts (lower is better)

### Quality Indicators
- **Geometric Stress**: Target < 1.0 (critical if > 5.0)
- **Quantum Purity**: Healthy if > 0.95
- **Guard Triggers** (Phase 2): Alarm if > 100
- **Execution Time**: Tracked for performance analysis

### Phase-Specific Metrics
- **Phase 2 (Thermodynamic)**:
  - Guard triggers (conflict escalations)
  - Compaction ratio (color compression)
  - Acceptance rate (annealing efficiency)

- **Phase 3 (Quantum)**:
  - Purity (coherence measure)
  - Final color count (post-evolution)
  - Conflicts remaining

- **Conflict Repair**:
  - Color increase (how many colors added)
  - Conflicts eliminated
  - Success rate

## Campaign Directory Structure

```
campaigns/world_record_20250123_140530/
├── state.json                      # Campaign state (for resume)
├── CHAMPION.toml                   # Best config found
├── CAMPAIGN_SUMMARY.md             # Auto-generated summary
├── configs/
│   ├── base.toml                   # Original config
│   ├── iter_1.toml                 # μ=0.50, coupling=7.0
│   ├── iter_2.toml                 # μ=0.55, coupling=7.0
│   └── ...
├── telemetry/
│   ├── iter_1.jsonl               # Full telemetry
│   ├── iter_2.jsonl
│   └── ...
├── results/
│   ├── iter_1.log                 # PRISM stdout/stderr
│   ├── iter_2.log
│   └── ...
├── analysis/
│   ├── campaign_results.csv       # All metrics in CSV
│   ├── statistical_report.txt     # Statistical summary
│   ├── phase_analysis.txt         # Phase breakdown
│   ├── recommendations.txt        # Automated recommendations
│   ├── plot_results.py            # Visualization script
│   ├── convergence.png            # Plots (if generated)
│   ├── parameter_exploration.png
│   └── phase_analysis.png
└── build_iter_*.log               # GPU compilation logs
```

## Resuming a Campaign

If a campaign is interrupted:
```bash
# Resume from last completed iteration
./scripts/world_record_campaign.sh world_record_20250123_140530
```

The script will:
- Load state from `state.json`
- Continue from last completed iteration
- Preserve best result tracking
- Generate updated summary

## Target Achievement

When **17 colors + 0 conflicts** is achieved:

1. Campaign stops immediately
2. Best config saved as `CHAMPION.toml`
3. Summary report generated
4. Success banner displayed

**To verify:**
```bash
cargo run --release --features=cuda -- \
    --config campaigns/world_record_20250123_140530/CHAMPION.toml \
    --graph data/graphs/DSJC500.5.col \
    --telemetry verification.jsonl
```

## Troubleshooting

### GPU Kernel Compilation Fails

**Issue:** Chemical potential change requires recompilation, but build fails.

**Solution:**
```bash
cd prism-gpu
# Check CUDA is available
nvcc --version

# Clean and rebuild
cargo clean
cargo build --release --features=cuda

# Check for errors
cat ../campaigns/<campaign_name>/build_iter_*.log
```

### Campaign Runs Slowly

**Issue:** Each iteration takes 5+ minutes.

**Solutions:**
- Ensure GPU features enabled: `--features=cuda`
- Check GPU utilization: `nvidia-smi`
- Reduce quantum evolution_iterations in base config
- Use smaller test graph initially (DSJC125.5)

### All Runs Fail (chromatic = 999)

**Issue:** PRISM crashes or times out on every iteration.

**Solution:**
```bash
# Check most recent log
tail -50 campaigns/<campaign_name>/results/iter_1.log

# Run single iteration manually to diagnose
cargo run --release --features=cuda -- \
    --config campaigns/<campaign_name>/configs/iter_1.toml \
    --graph data/graphs/DSJC500.5.col \
    --telemetry test.jsonl
```

### Monitor Shows No Updates

**Issue:** Monitor script shows old data or "In Progress" forever.

**Solution:**
- Check if campaign script is still running: `ps aux | grep world_record_campaign`
- Verify telemetry files being created: `ls -ltr campaigns/<campaign_name>/telemetry/`
- Increase refresh interval: `./scripts/monitor_campaign.sh <dir> 10`

## Parameter Tuning Tips

### If Getting High Conflicts (> 50)

**Reduce compression aggression:**
- Decrease chemical potential μ (try 0.50-0.55)
- Widen thermodynamic temperature range
- Increase cooling steps

### If Stuck Above Target Chromatic (> 17)

**Increase compression:**
- Increase chemical potential μ (try 0.65-0.70)
- Increase quantum coupling strength (8.5-9.0)
- Extend evolution time (0.16-0.18)
- Increase memetic generations

### If High Geometric Stress (> 5.0)

**Improve phase coordination:**
- Reduce metaphysical feedback_strength
- Increase ensemble diversity_weight
- Balance warmstart ratios
- Increase stress_decay_rate

## Advanced Usage

### Multi-Fidelity Campaign

Start with small graph, scale to DSJC500.5:
```bash
# Phase 1: Quick exploration on DSJC125.5
./scripts/world_record_campaign.sh quick_test \
    configs/WORLD_RECORD_ATTEMPT.toml \
    data/graphs/DSJC125.5.col

# Analyze to find best parameters
./scripts/analyze_campaign_results.sh campaigns/quick_test

# Phase 2: Refine on DSJC500.5 with learned parameters
# (edit configs/REFINED.toml with recommended parameters)
./scripts/world_record_campaign.sh production \
    configs/REFINED.toml \
    data/graphs/DSJC500.5.col
```

### Parallel Campaign Testing

Run multiple campaigns with different strategies:
```bash
# Terminal 1: Conservative (low μ)
./scripts/world_record_campaign.sh conservative configs/LOW_MU.toml

# Terminal 2: Aggressive (high μ)
./scripts/world_record_campaign.sh aggressive configs/HIGH_MU.toml

# Terminal 3: Balanced
./scripts/world_record_campaign.sh balanced configs/WORLD_RECORD_ATTEMPT.toml

# Monitor all
./scripts/monitor_campaign.sh campaigns/conservative &
./scripts/monitor_campaign.sh campaigns/aggressive &
./scripts/monitor_campaign.sh campaigns/balanced
```

### Custom Parameter Ranges

Edit the campaign script arrays:
```bash
# In world_record_campaign.sh, modify:
declare -a MU_VALUES=(0.55 0.60 0.65)           # Narrower range
declare -a COUPLING_VALUES=(8.0 8.5 9.0 9.5)   # Higher values
declare -a EVOLUTION_TIME_VALUES=(0.15 0.16 0.17 0.18)  # Longer times
```

## Integration with CI/CD

### Automated Nightly Campaigns

```bash
#!/bin/bash
# nightly_hypertune.sh

DATE=$(date +%Y%m%d)
CAMPAIGN_NAME="nightly_$DATE"

cd /home/user/PRISM
./scripts/world_record_campaign.sh "$CAMPAIGN_NAME" > "logs/campaign_$DATE.log" 2>&1

# Analyze results
./scripts/analyze_campaign_results.sh "campaigns/$CAMPAIGN_NAME"

# Email/Slack notification if target achieved
if jq -e '.target_achieved == true' "campaigns/$CAMPAIGN_NAME/state.json"; then
    echo "🏆 World record achieved on $DATE!" | mail -s "PRISM Success" team@example.com
fi
```

### GitHub Actions Integration

```yaml
name: PRISM Hypertuning Campaign

on:
  schedule:
    - cron: '0 2 * * *'  # 2 AM daily
  workflow_dispatch:

jobs:
  hypertune:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v3
      - name: Run Campaign
        run: |
          ./scripts/world_record_campaign.sh "github_${{ github.run_number }}"
      - name: Analyze Results
        run: |
          ./scripts/analyze_campaign_results.sh "campaigns/github_${{ github.run_number }}"
      - name: Upload Artifacts
        uses: actions/upload-artifact@v3
        with:
          name: campaign-results
          path: campaigns/github_${{ github.run_number }}/analysis/
```

## Next Steps

1. **Start Your First Campaign:**
   ```bash
   ./scripts/world_record_campaign.sh
   ```

2. **Monitor in Real-Time:**
   ```bash
   # New terminal
   ./scripts/monitor_campaign.sh
   ```

3. **Analyze and Iterate:**
   ```bash
   ./scripts/analyze_campaign_results.sh
   # Review recommendations.txt
   # Adjust parameters and run again
   ```

4. **When Target Achieved:**
   - Verify with multiple runs
   - Test on other graphs
   - Submit to DIMACS benchmarks
   - Document in paper/report

## Support

For issues or questions:
- Check telemetry logs in `campaigns/*/telemetry/`
- Review PRISM logs in `campaigns/*/results/`
- Inspect GPU compilation logs: `campaigns/*/build_iter_*.log`
- See recommendations in `campaigns/*/analysis/recommendations.txt`

## Success Criteria

Campaign is successful when:
- ✅ Chromatic number ≤ 17
- ✅ Conflicts = 0
- ✅ Geometric stress < 1.0
- ✅ Reproducible across multiple runs
- ✅ Telemetry shows stable convergence

**Good luck achieving the world record! 🏆**
