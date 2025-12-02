#!/usr/bin/env python3
"""
External Ground Truth Benchmark Evaluation for PRISM-LBS

Uses LIGYSIS-style evaluation methodology with EXTERNAL ground truth sources:
- Tier 1: 10 PDB co-crystal structures (classic drug binding sites)
- CryptoSite: 18 structures (cryptic binding site benchmark)
- ASBench: 15 structures (allosteric site benchmark)
- Novel Targets: 10 structures (historically "undruggable" targets)

Total: 53 structures with validated external ground truth

Ground Truth Sources:
- PDB: Crystal structures with bound ligands
- CryptoSite Paper: Cimermancic et al. J Mol Biol 2016
- ASBench: Huang et al. Nucleic Acids Res 2015
- Novel targets: FDA-approved drug binding sites

Metrics:
- Top-N+K recall: For protein with N sites, check if top N+K predictions find all sites
- Relative Intersection (IRel): |pred ∩ ref| / |ref|
- Per-tier and overall recall statistics
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
from scipy.spatial.distance import euclidean

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
GROUND_TRUTH_DIR = PROJECT_ROOT / "benchmark" / "complete_validation" / "ground_truth"


@dataclass
class BindingSite:
    """Represents a binding site (ground truth or prediction)."""
    site_id: int
    pdb_id: str
    residues: Set[str]
    centroid: Tuple[float, float, float]
    score: float = 0.0
    rank: int = 0
    tier: str = ""
    site_type: str = ""


def load_tier1_ground_truth(csv_path: Path) -> Dict[str, List[BindingSite]]:
    """Load Tier 1 (Table Stakes) ground truth."""
    sites_by_pdb = defaultdict(list)

    try:
        df = pd.read_csv(csv_path, comment='#')
        for _, row in df.iterrows():
            pdb_id = str(row['pdb_id']).lower()
            residues_str = str(row.get('binding_residues', ''))
            residues = {str(r).strip() for r in residues_str.split(';') if r.strip()}

            sites_by_pdb[pdb_id].append(BindingSite(
                site_id=len(sites_by_pdb[pdb_id]) + 1,
                pdb_id=pdb_id,
                residues=residues,
                centroid=(0.0, 0.0, 0.0),  # Will use residue overlap
                score=1.0,
                rank=1,
                tier="tier1",
                site_type=str(row.get('site_type', 'active_site'))
            ))
    except Exception as e:
        print(f"Error loading Tier 1: {e}")

    return dict(sites_by_pdb)


def load_cryptosite_ground_truth(csv_path: Path) -> Dict[str, List[BindingSite]]:
    """Load CryptoSite (cryptic binding sites) ground truth."""
    sites_by_pdb = defaultdict(list)

    try:
        df = pd.read_csv(csv_path, comment='#')
        for _, row in df.iterrows():
            pdb_id = str(row['pdb_id']).lower()
            residues_str = str(row.get('cryptic_residues', ''))
            residues = {str(r).strip() for r in residues_str.split(';') if r.strip()}

            sites_by_pdb[pdb_id].append(BindingSite(
                site_id=len(sites_by_pdb[pdb_id]) + 1,
                pdb_id=pdb_id,
                residues=residues,
                centroid=(0.0, 0.0, 0.0),
                score=1.0,
                rank=1,
                tier="cryptosite",
                site_type=str(row.get('mechanism', 'cryptic'))
            ))
    except Exception as e:
        print(f"Error loading CryptoSite: {e}")

    return dict(sites_by_pdb)


def load_asbench_ground_truth(csv_path: Path) -> Dict[str, List[BindingSite]]:
    """Load ASBench (allosteric sites) ground truth."""
    sites_by_pdb = defaultdict(list)

    try:
        df = pd.read_csv(csv_path, comment='#')
        for _, row in df.iterrows():
            pdb_id = str(row['pdb_id']).lower()

            # Load allosteric site residues
            allo_residues_str = str(row.get('allosteric_residues', ''))
            allo_residues = {str(r).strip() for r in allo_residues_str.split(';') if r.strip()}

            if allo_residues:
                sites_by_pdb[pdb_id].append(BindingSite(
                    site_id=len(sites_by_pdb[pdb_id]) + 1,
                    pdb_id=pdb_id,
                    residues=allo_residues,
                    centroid=(0.0, 0.0, 0.0),
                    score=1.0,
                    rank=1,
                    tier="asbench",
                    site_type=str(row.get('mechanism', 'allosteric'))
                ))
    except Exception as e:
        print(f"Error loading ASBench: {e}")

    return dict(sites_by_pdb)


def load_novel_targets_ground_truth(csv_path: Path) -> Dict[str, List[BindingSite]]:
    """Load Novel Targets (undruggable) ground truth."""
    sites_by_pdb = defaultdict(list)

    try:
        df = pd.read_csv(csv_path, comment='#')
        for _, row in df.iterrows():
            pdb_id = str(row['pdb_id']).lower()
            residues_str = str(row.get('site_residues', ''))
            residues = {str(r).strip() for r in residues_str.split(';') if r.strip()}

            sites_by_pdb[pdb_id].append(BindingSite(
                site_id=len(sites_by_pdb[pdb_id]) + 1,
                pdb_id=pdb_id,
                residues=residues,
                centroid=(0.0, 0.0, 0.0),
                score=1.0,
                rank=1,
                tier="novel",
                site_type=str(row.get('site_type', 'novel'))
            ))
    except Exception as e:
        print(f"Error loading Novel Targets: {e}")

    return dict(sites_by_pdb)


def load_all_ground_truth() -> Tuple[Dict[str, List[BindingSite]], Dict[str, str]]:
    """Load all external ground truth sources."""
    all_sites = {}
    pdb_to_tier = {}

    # Tier 1
    tier1_path = GROUND_TRUTH_DIR / "tier1_binding_sites.csv"
    if tier1_path.exists():
        tier1 = load_tier1_ground_truth(tier1_path)
        for pdb_id, sites in tier1.items():
            all_sites[pdb_id] = sites
            pdb_to_tier[pdb_id] = "Tier1"
        print(f"  Tier 1 (Table Stakes): {len(tier1)} structures")

    # CryptoSite
    crypto_path = GROUND_TRUTH_DIR / "cryptosite_ground_truth.csv"
    if crypto_path.exists():
        crypto = load_cryptosite_ground_truth(crypto_path)
        for pdb_id, sites in crypto.items():
            if pdb_id not in all_sites:
                all_sites[pdb_id] = sites
                pdb_to_tier[pdb_id] = "CryptoSite"
        print(f"  CryptoSite (Cryptic):  {len(crypto)} structures")

    # ASBench
    asbench_path = GROUND_TRUTH_DIR / "asbench_ground_truth.csv"
    if asbench_path.exists():
        asbench = load_asbench_ground_truth(asbench_path)
        for pdb_id, sites in asbench.items():
            if pdb_id not in all_sites:
                all_sites[pdb_id] = sites
                pdb_to_tier[pdb_id] = "ASBench"
        print(f"  ASBench (Allosteric):  {len(asbench)} structures")

    # Novel targets
    novel_path = GROUND_TRUTH_DIR / "novel_targets_ground_truth.csv"
    if novel_path.exists():
        novel = load_novel_targets_ground_truth(novel_path)
        for pdb_id, sites in novel.items():
            if pdb_id not in all_sites:
                all_sites[pdb_id] = sites
                pdb_to_tier[pdb_id] = "Novel"
        print(f"  Novel Targets:         {len(novel)} structures")

    return all_sites, pdb_to_tier


def parse_prism_json(json_path: Path) -> List[BindingSite]:
    """Parse PRISM publication JSON output."""
    sites = []

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        return []

    pdb_id = json_path.stem.lower()

    for i, pocket in enumerate(data.get('pockets', [])):
        # Handle both old and new format
        if 'residue_indices' in pocket:
            residues = {str(r) for r in pocket['residue_indices']}
        elif 'residues' in pocket:
            residues = {str(r.get('id', r)) for r in pocket['residues']}
        else:
            residues = set()

        # Get centroid
        geometry = pocket.get('geometry', {})
        centroid = tuple(geometry.get('centroid', [0, 0, 0]))

        # Get score
        scores = pocket.get('scores', {})
        score = scores.get('confidence', scores.get('druggability_total', 0.0))

        sites.append(BindingSite(
            site_id=pocket.get('id', i + 1),
            pdb_id=pdb_id,
            residues=residues,
            centroid=centroid,
            score=score,
            rank=pocket.get('rank', i + 1),
            tier="prediction",
            site_type="predicted"
        ))

    return sites


def compute_relative_intersection(pred_residues: Set[str], ref_residues: Set[str]) -> float:
    """Compute IRel: |pred ∩ ref| / |ref|"""
    if not ref_residues:
        return 0.0
    intersection = len(pred_residues & ref_residues)
    return intersection / len(ref_residues)


def get_best_prediction_for_site(
    ref_site: BindingSite,
    predictions: List[BindingSite],
    rank_limit: Optional[int] = None
) -> Tuple[Optional[BindingSite], float]:
    """Find the best matching prediction for a reference site."""
    best_pred = None
    best_irel = 0.0

    for pred in predictions:
        if rank_limit is not None and pred.rank > rank_limit:
            continue

        irel = compute_relative_intersection(pred.residues, ref_site.residues)
        if irel > best_irel:
            best_pred = pred
            best_irel = irel

    return best_pred, best_irel


def evaluate_topn_plus_k(
    reference: Dict[str, List[BindingSite]],
    predictions: Dict[str, List[BindingSite]],
    k: int = 2,
    irel_threshold: float = 0.25
) -> Dict:
    """
    LIGYSIS-style Top-N+K recall evaluation.

    For each protein with N binding sites, check if top N+K predictions
    contain all N sites with IRel >= threshold.
    """
    total_sites = 0
    correct_sites = 0
    per_structure = []

    for pdb_id, ref_sites in reference.items():
        n_sites = len(ref_sites)
        if n_sites == 0:
            continue

        preds = predictions.get(pdb_id, [])
        rank_limit = n_sites + k

        struct_correct = 0
        for ref_site in ref_sites:
            total_sites += 1
            _, best_irel = get_best_prediction_for_site(ref_site, preds, rank_limit)

            if best_irel >= irel_threshold:
                correct_sites += 1
                struct_correct += 1

        per_structure.append({
            'pdb_id': pdb_id,
            'n_sites': n_sites,
            'rank_limit': rank_limit,
            'correct': struct_correct,
            'recall': struct_correct / n_sites if n_sites > 0 else 0
        })

    recall = correct_sites / total_sites if total_sites > 0 else 0
    se = np.sqrt(recall * (1 - recall) / total_sites) if total_sites > 0 else 0

    return {
        'recall': recall,
        'recall_pct': recall * 100,
        'correct': correct_sites,
        'total': total_sites,
        'ci_lower': max(0, recall - 1.96 * se) * 100,
        'ci_upper': min(1, recall + 1.96 * se) * 100,
        'k': k,
        'irel_threshold': irel_threshold,
        'per_structure': per_structure
    }


def run_evaluation(json_dir: Path, verbose: bool = False) -> pd.DataFrame:
    """Run full LIGYSIS-style evaluation on external ground truth."""

    print("\n" + "=" * 70)
    print("PRISM-LBS External Ground Truth Benchmark")
    print("=" * 70)
    print("\nLoading ground truth from external sources:")

    # Load all ground truth
    reference, pdb_to_tier = load_all_ground_truth()

    if not reference:
        print("\nERROR: No ground truth found!")
        return pd.DataFrame()

    total_sites = sum(len(s) for s in reference.values())
    print(f"\nTotal: {len(reference)} structures, {total_sites} binding sites")

    # Load predictions
    print(f"\nLoading PRISM predictions from: {json_dir}")
    predictions = {}
    json_files = list(json_dir.glob('*.json'))

    for json_file in json_files:
        sites = parse_prism_json(json_file)
        if sites:
            pdb_id = json_file.stem.lower()
            predictions[pdb_id] = sites

    print(f"Loaded predictions for {len(predictions)} structures")

    # Find common structures
    common = set(reference.keys()) & set(predictions.keys())
    print(f"Evaluating on {len(common)} common structures\n")

    if not common:
        print("ERROR: No common structures between ground truth and predictions!")
        print(f"Ground truth PDBs: {list(reference.keys())[:10]}...")
        print(f"Prediction PDBs: {list(predictions.keys())[:10]}...")
        return pd.DataFrame()

    # Filter to common structures
    ref_filtered = {k: v for k, v in reference.items() if k in common}
    pred_filtered = {k: v for k, v in predictions.items() if k in common}

    # Run evaluation for different k values and thresholds
    k_values = [0, 1, 2, 3]
    irel_thresholds = [0.25, 0.30, 0.40, 0.50]

    results = []
    for k in k_values:
        for irel_t in irel_thresholds:
            result = evaluate_topn_plus_k(
                ref_filtered, pred_filtered, k=k, irel_threshold=irel_t
            )
            results.append({
                'method': 'PRISM-LBS',
                'k': k,
                'rank_metric': f'Top-N+{k}',
                'irel_threshold': irel_t,
                'recall_pct': result['recall_pct'],
                'correct': result['correct'],
                'total': result['total'],
                'ci_lower': result['ci_lower'],
                'ci_upper': result['ci_upper']
            })

    df = pd.DataFrame(results)

    # Print results
    print_results(df, pdb_to_tier, ref_filtered, pred_filtered)

    # Per-tier breakdown
    if verbose:
        print_tier_breakdown(ref_filtered, pred_filtered, pdb_to_tier)

    return df


def print_results(df: pd.DataFrame, pdb_to_tier: Dict, ref: Dict, pred: Dict):
    """Print formatted results."""

    # LIGYSIS paper baselines
    baselines = {
        'fpocket': 60,
        'P2Rank': 54,
        'DeepPocket': 60,
        'VN-EGNN': 46,
        'PUResNet': 45,
        'IF-SitePred': 55,
    }

    print("\n" + "=" * 70)
    print("RESULTS: PRISM-LBS vs LIGYSIS Baselines")
    print("=" * 70)

    # Primary metric: Top-N+2 @ IRel=0.25
    row = df[(df['k'] == 2) & (df['irel_threshold'] == 0.25)]
    if not row.empty:
        prism_recall = row.iloc[0]['recall_pct']
        ci_low = row.iloc[0]['ci_lower']
        ci_high = row.iloc[0]['ci_upper']
    else:
        prism_recall = 0
        ci_low = ci_high = 0

    print(f"\nPrimary Metric: Top-N+2 Recall @ IRel=0.25")
    print("-" * 50)
    print(f"{'Method':<20} {'Recall (%)':<15} {'vs PRISM'}")
    print("-" * 50)
    print(f"{'PRISM-LBS':<20} {prism_recall:>5.1f} [{ci_low:.0f}-{ci_high:.0f}]")

    for method, recall in sorted(baselines.items(), key=lambda x: -x[1]):
        diff = prism_recall - recall
        sign = '+' if diff > 0 else ''
        print(f"{method:<20} {recall:>5.1f}             {sign}{diff:.1f}")

    print("-" * 50)

    # Target assessment
    target = 60  # fpocket baseline
    if prism_recall >= target:
        print(f"\n✓ PASSED: PRISM-LBS achieves {prism_recall:.1f}% (≥{target}% target)")
    else:
        gap = target - prism_recall
        print(f"\n→ PENDING: PRISM-LBS at {prism_recall:.1f}% (need +{gap:.1f}% for {target}% target)")

    # Full results matrix
    print("\n\nFull Results Matrix:")
    print("-" * 70)
    pivot = df.pivot_table(
        values='recall_pct',
        index='irel_threshold',
        columns='rank_metric',
        aggfunc='first'
    )
    print(pivot.round(1).to_string())


def print_tier_breakdown(ref: Dict, pred: Dict, pdb_to_tier: Dict):
    """Print per-tier performance breakdown."""

    print("\n\nPer-Tier Breakdown:")
    print("-" * 70)

    tiers = {'Tier1': [], 'CryptoSite': [], 'ASBench': [], 'Novel': []}

    for pdb_id in ref.keys():
        tier = pdb_to_tier.get(pdb_id, 'Unknown')
        if tier in tiers:
            tiers[tier].append(pdb_id)

    for tier_name, pdb_ids in tiers.items():
        if not pdb_ids:
            continue

        tier_ref = {k: v for k, v in ref.items() if k in pdb_ids}
        tier_pred = {k: v for k, v in pred.items() if k in pdb_ids}

        result = evaluate_topn_plus_k(tier_ref, tier_pred, k=2, irel_threshold=0.25)

        print(f"{tier_name:<15} {result['correct']:>3}/{result['total']:<3} = {result['recall_pct']:>5.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description='External Ground Truth Benchmark for PRISM-LBS'
    )
    parser.add_argument(
        '--json-dir',
        type=Path,
        required=True,
        help='Directory containing PRISM JSON predictions'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output CSV file for results'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show per-tier breakdown'
    )

    args = parser.parse_args()

    if not args.json_dir.exists():
        print(f"Error: JSON directory not found: {args.json_dir}")
        sys.exit(1)

    df = run_evaluation(args.json_dir, verbose=args.verbose)

    if args.output and not df.empty:
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")

    return df


if __name__ == '__main__':
    main()
