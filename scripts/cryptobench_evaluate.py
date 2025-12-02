#!/usr/bin/env python3
"""
CryptoBench External Ground Truth Evaluation for PRISM-LBS

Uses the CryptoBench dataset (Bioinformatics 2024) with 1,107 APO structures
and 5,493 cryptic binding site annotations.

External Ground Truth Source:
- OSF: https://osf.io/pz4a9/
- Paper: https://academic.oup.com/bioinformatics/article/41/1/btae745/7927823
- GitHub: https://github.com/skrhakv/CryptoBench

Evaluation Metrics:
- Top-N+K recall (LIGYSIS-style)
- Relative Intersection (IRel): |pred ∩ ref| / |ref|
- DCC (Distance from Centroid to Closest reference site)
"""

import os
import sys
import json
import urllib.request
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import subprocess
import tempfile
import shutil

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
CRYPTOBENCH_DIR = PROJECT_ROOT / "benchmark" / "cryptobench"


@dataclass
class CrypticSite:
    """Represents a cryptic binding site from CryptoBench."""
    apo_pdb: str
    apo_chain: str
    holo_pdb: str
    ligand: str
    residues: Set[str]  # Just residue numbers
    pRMSD: float
    is_main: bool


@dataclass
class Prediction:
    """Represents a PRISM prediction."""
    pdb_id: str
    pocket_id: int
    residues: Set[str]
    score: float
    rank: int


def load_cryptobench_ground_truth(
    dataset_path: Path,
    folds_path: Path,
    split: str = "test"
) -> Dict[str, List[CrypticSite]]:
    """
    Load CryptoBench ground truth for specified split.

    Returns dict mapping apo_pdb -> list of CrypticSite
    """
    with open(dataset_path) as f:
        dataset = json.load(f)

    with open(folds_path) as f:
        folds = json.load(f)

    # Get APO IDs for this split
    split_apos = set(folds.get(split, []))

    ground_truth = defaultdict(list)

    for apo_pdb, holos in dataset.items():
        if apo_pdb not in split_apos:
            continue

        for holo in holos:
            # Parse residues from apo_pocket_selection
            # Format: ["B_12", "B_14", ...] -> just the residue numbers
            apo_chain = holo.get('apo_chain', 'A')
            residues = set()
            for res in holo.get('apo_pocket_selection', []):
                if '_' in res:
                    chain, resnum = res.split('_', 1)
                    if chain == apo_chain:
                        residues.add(resnum)
                else:
                    residues.add(res)

            if residues:
                ground_truth[apo_pdb].append(CrypticSite(
                    apo_pdb=apo_pdb,
                    apo_chain=apo_chain,
                    holo_pdb=holo.get('holo_pdb_id', ''),
                    ligand=holo.get('ligand', ''),
                    residues=residues,
                    pRMSD=holo.get('pRMSD', 0.0),
                    is_main=holo.get('is_main_holo_structure', False)
                ))

    return dict(ground_truth)


def download_pdb(pdb_id: str, output_dir: Path, use_cif: bool = True) -> Optional[Path]:
    """Download PDB/CIF file from RCSB."""
    pdb_id = pdb_id.lower()

    if use_cif:
        url = f"https://files.rcsb.org/download/{pdb_id}.cif"
        output_path = output_dir / f"{pdb_id}.cif"
    else:
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        output_path = output_dir / f"{pdb_id}.pdb"

    if output_path.exists():
        return output_path

    try:
        urllib.request.urlretrieve(url, output_path)
        return output_path
    except Exception as e:
        # Try alternate format
        if use_cif:
            return download_pdb(pdb_id, output_dir, use_cif=False)
        print(f"  Failed to download {pdb_id}: {e}")
        return None


def parse_prism_json(json_path: Path) -> List[Prediction]:
    """Parse PRISM JSON output."""
    predictions = []

    try:
        with open(json_path) as f:
            data = json.load(f)
    except Exception:
        return []

    pdb_id = json_path.stem.lower()

    for i, pocket in enumerate(data.get('pockets', [])):
        # Extract residue numbers
        residues = set()
        if 'residue_indices' in pocket:
            residues = {str(r) for r in pocket['residue_indices']}

        scores = pocket.get('scores', {})
        score = scores.get('confidence', scores.get('druggability_total', 0.0))

        predictions.append(Prediction(
            pdb_id=pdb_id,
            pocket_id=pocket.get('id', i + 1),
            residues=residues,
            score=score,
            rank=pocket.get('rank', i + 1)
        ))

    return predictions


def compute_irel(pred_residues: Set[str], ref_residues: Set[str]) -> float:
    """Compute relative intersection: |pred ∩ ref| / |ref|"""
    if not ref_residues:
        return 0.0
    intersection = len(pred_residues & ref_residues)
    return intersection / len(ref_residues)


def evaluate_structure(
    sites: List[CrypticSite],
    predictions: List[Prediction],
    k: int = 2,
    irel_threshold: float = 0.25
) -> Dict:
    """
    Evaluate predictions against ground truth sites.

    For cryptic sites, we use Top-N+K where N = number of unique sites
    (after merging sites with high overlap).
    """
    if not sites or not predictions:
        return {'hits': 0, 'total': 0, 'recall': 0.0}

    # Merge overlapping sites (same binding region from different holos)
    merged_sites = merge_sites(sites)
    n_sites = len(merged_sites)
    rank_limit = n_sites + k

    hits = 0
    for site in merged_sites:
        # Find best matching prediction within rank limit
        best_irel = 0.0
        for pred in predictions:
            if pred.rank > rank_limit:
                continue
            irel = compute_irel(pred.residues, site.residues)
            best_irel = max(best_irel, irel)

        if best_irel >= irel_threshold:
            hits += 1

    return {
        'hits': hits,
        'total': n_sites,
        'recall': hits / n_sites if n_sites > 0 else 0.0
    }


def merge_sites(sites: List[CrypticSite], jaccard_threshold: float = 0.5) -> List[CrypticSite]:
    """Merge sites with high residue overlap."""
    if len(sites) <= 1:
        return sites

    merged = []
    used = set()

    for i, site1 in enumerate(sites):
        if i in used:
            continue

        merged_residues = set(site1.residues)

        for j, site2 in enumerate(sites[i+1:], start=i+1):
            if j in used:
                continue

            intersection = len(site1.residues & site2.residues)
            union = len(site1.residues | site2.residues)
            jaccard = intersection / union if union > 0 else 0

            if jaccard >= jaccard_threshold:
                merged_residues |= site2.residues
                used.add(j)

        merged.append(CrypticSite(
            apo_pdb=site1.apo_pdb,
            apo_chain=site1.apo_chain,
            holo_pdb=site1.holo_pdb,
            ligand=site1.ligand,
            residues=merged_residues,
            pRMSD=site1.pRMSD,
            is_main=site1.is_main
        ))

    return merged


def compute_classification_metrics(
    ground_truth: Dict[str, List[CrypticSite]],
    predictions_by_pdb: Dict[str, List[Prediction]],
    irel_threshold: float = 0.25
) -> Dict:
    """
    Compute classification metrics (AUC, AUPRC, MCC, F1, TPR, FPR).

    For each predicted pocket, classify as TP/FP based on whether it
    overlaps with any ground truth site at >= irel_threshold.
    """
    y_true = []  # 1 if prediction matches ground truth, 0 otherwise
    y_scores = []  # prediction confidence scores

    common = set(ground_truth.keys()) & set(predictions_by_pdb.keys())

    for pdb_id in common:
        sites = ground_truth[pdb_id]
        preds = predictions_by_pdb[pdb_id]

        # Merge overlapping ground truth sites
        merged_sites = merge_sites(sites)
        all_gt_residues = set()
        for site in merged_sites:
            all_gt_residues |= site.residues

        for pred in preds:
            # Check if this prediction overlaps with any ground truth site
            is_hit = False
            for site in merged_sites:
                irel = compute_irel(pred.residues, site.residues)
                if irel >= irel_threshold:
                    is_hit = True
                    break

            y_true.append(1 if is_hit else 0)
            y_scores.append(pred.score)

    if not y_true or sum(y_true) == 0:
        return {'auc': 0.0, 'auprc': 0.0, 'mcc': 0.0, 'f1': 0.0, 'tpr': 0.0, 'fpr': 0.0}

    # Compute metrics
    from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, f1_score

    # Binary predictions at threshold 0.5
    y_pred = [1 if s >= 0.5 else 0 for s in y_scores]

    try:
        auc = roc_auc_score(y_true, y_scores)
    except:
        auc = 0.0

    try:
        auprc = average_precision_score(y_true, y_scores)
    except:
        auprc = 0.0

    try:
        mcc = matthews_corrcoef(y_true, y_pred)
    except:
        mcc = 0.0

    try:
        f1 = f1_score(y_true, y_pred)
    except:
        f1 = 0.0

    # TPR and FPR
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        'auc': auc,
        'auprc': auprc,
        'mcc': mcc,
        'f1': f1,
        'tpr': tpr,
        'fpr': fpr,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
    }


def run_evaluation(
    ground_truth: Dict[str, List[CrypticSite]],
    predictions_dir: Path,
    k_values: List[int] = [0, 1, 2, 3],
    irel_thresholds: List[float] = [0.25, 0.30, 0.40, 0.50]
) -> Dict:
    """Run full evaluation."""

    results = defaultdict(lambda: {'hits': 0, 'total': 0})
    per_structure = []

    # Load all predictions
    predictions_by_pdb = {}
    for json_file in predictions_dir.glob('*.json'):
        pdb_id = json_file.stem.lower()
        predictions_by_pdb[pdb_id] = parse_prism_json(json_file)

    # Evaluate each structure
    common = set(ground_truth.keys()) & set(predictions_by_pdb.keys())

    for pdb_id in common:
        sites = ground_truth[pdb_id]
        preds = predictions_by_pdb[pdb_id]

        for k in k_values:
            for irel_t in irel_thresholds:
                result = evaluate_structure(sites, preds, k=k, irel_threshold=irel_t)
                key = (k, irel_t)
                results[key]['hits'] += result['hits']
                results[key]['total'] += result['total']

        # Store per-structure results for k=2, irel=0.25
        result = evaluate_structure(sites, preds, k=2, irel_threshold=0.25)
        per_structure.append({
            'pdb_id': pdb_id,
            'hits': result['hits'],
            'total': result['total'],
            'recall': result['recall']
        })

    # Compute final metrics
    final_results = []
    for (k, irel_t), counts in results.items():
        recall = counts['hits'] / counts['total'] if counts['total'] > 0 else 0
        se = np.sqrt(recall * (1 - recall) / counts['total']) if counts['total'] > 0 else 0

        final_results.append({
            'k': k,
            'irel_threshold': irel_t,
            'rank_metric': f'Top-N+{k}',
            'recall_pct': recall * 100,
            'hits': counts['hits'],
            'total': counts['total'],
            'ci_lower': max(0, recall - 1.96 * se) * 100,
            'ci_upper': min(1, recall + 1.96 * se) * 100
        })

    # Compute classification metrics (AUC, AUPRC, MCC, F1, TPR, FPR)
    try:
        classification_metrics = compute_classification_metrics(
            ground_truth, predictions_by_pdb, irel_threshold=0.25
        )
    except ImportError:
        classification_metrics = {
            'auc': 'N/A (sklearn required)',
            'auprc': 'N/A',
            'mcc': 'N/A',
            'f1': 'N/A',
            'tpr': 'N/A',
            'fpr': 'N/A'
        }

    return {
        'results': final_results,
        'classification': classification_metrics,
        'per_structure': per_structure,
        'n_common': len(common),
        'n_ground_truth': len(ground_truth),
        'n_predictions': len(predictions_by_pdb)
    }


def print_results(eval_results: Dict):
    """Print formatted results."""

    print("\n" + "=" * 70)
    print("CryptoBench External Ground Truth Benchmark Results")
    print("=" * 70)
    print(f"\nDataset: CryptoBench (OSF: https://osf.io/pz4a9/)")
    print(f"Ground truth structures: {eval_results['n_ground_truth']}")
    print(f"Predictions loaded: {eval_results['n_predictions']}")
    print(f"Common structures evaluated: {eval_results['n_common']}")

    # Primary metrics (AUC, AUPRC)
    classification = eval_results.get('classification', {})
    if classification and isinstance(classification.get('auc'), float):
        print(f"\n" + "-" * 50)
        print(f"PRIMARY METRICS (Classification)")
        print(f"-" * 50)
        print(f"  AUC:    {classification['auc']:.3f}")
        print(f"  AUPRC:  {classification['auprc']:.3f}")

        print(f"\nSECONDARY METRICS")
        print(f"-" * 50)
        print(f"  MCC:    {classification['mcc']:.3f}")
        print(f"  F1:     {classification['f1']:.3f}")
        print(f"  TPR:    {classification['tpr']:.3f}")
        print(f"  FPR:    {classification['fpr']:.3f}")

        if 'tp' in classification:
            print(f"\nConfusion Matrix:")
            print(f"  TP: {classification['tp']:4}  FP: {classification['fp']:4}")
            print(f"  FN: {classification['fn']:4}  TN: {classification['tn']:4}")

    # Top-N+K recall
    results = eval_results['results']
    primary = next((r for r in results if r['k'] == 2 and r['irel_threshold'] == 0.25), None)

    if primary:
        print(f"\n" + "-" * 50)
        print(f"RECALL METRICS (Top-N+K)")
        print(f"-" * 50)
        print(f"Top-N+2 @ IRel=0.25: {primary['recall_pct']:.1f}% [{primary['ci_lower']:.0f}-{primary['ci_upper']:.0f}%]")
        print(f"Sites detected: {primary['hits']}/{primary['total']}")

    # Full matrix
    print(f"\n\nFull Recall Matrix:")
    print("-" * 70)
    print(f"{'IRel':<8}", end="")
    for k in [0, 1, 2, 3]:
        print(f"{'Top-N+' + str(k):<12}", end="")
    print()

    for irel in [0.25, 0.30, 0.40, 0.50]:
        print(f"{irel:<8.2f}", end="")
        for k in [0, 1, 2, 3]:
            r = next((x for x in results if x['k'] == k and x['irel_threshold'] == irel), None)
            if r:
                print(f"{r['recall_pct']:<12.1f}", end="")
            else:
                print(f"{'N/A':<12}", end="")
        print()


def main():
    parser = argparse.ArgumentParser(description='CryptoBench Evaluation for PRISM-LBS')
    parser.add_argument('--predictions', type=Path, required=True,
                        help='Directory containing PRISM JSON predictions')
    parser.add_argument('--split', type=str, default='test',
                        choices=['test', 'train-0', 'train-1', 'train-2', 'train-3'],
                        help='CryptoBench split to evaluate (default: test)')
    parser.add_argument('--output', type=Path, help='Output JSON file for results')
    parser.add_argument('--download-structures', action='store_true',
                        help='Download missing PDB structures')
    parser.add_argument('--structures-dir', type=Path,
                        help='Directory for PDB structures')

    args = parser.parse_args()

    # Check for dataset
    dataset_path = CRYPTOBENCH_DIR / 'dataset.json'
    folds_path = CRYPTOBENCH_DIR / 'folds.json'

    if not dataset_path.exists():
        print("ERROR: CryptoBench dataset not found!")
        print(f"Expected at: {dataset_path}")
        print("\nTo download, run:")
        print("  curl -sL 'https://osf.io/download/ta2ju/' -o benchmark/cryptobench/dataset.json")
        print("  curl -sL 'https://osf.io/download/5s93p/' -o benchmark/cryptobench/folds.json")
        sys.exit(1)

    # Load ground truth
    print(f"Loading CryptoBench ground truth (split: {args.split})...")
    ground_truth = load_cryptobench_ground_truth(dataset_path, folds_path, args.split)
    print(f"Loaded {len(ground_truth)} structures with cryptic sites")

    total_sites = sum(len(s) for s in ground_truth.values())
    print(f"Total cryptic site annotations: {total_sites}")

    # Optionally download structures
    if args.download_structures:
        structures_dir = args.structures_dir or (CRYPTOBENCH_DIR / 'structures' / args.split)
        structures_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nDownloading structures to {structures_dir}...")
        for pdb_id in ground_truth.keys():
            download_pdb(pdb_id, structures_dir)
        print("Download complete")

    # Run evaluation
    if not args.predictions.exists():
        print(f"ERROR: Predictions directory not found: {args.predictions}")
        sys.exit(1)

    print(f"\nRunning evaluation on {args.predictions}...")
    eval_results = run_evaluation(ground_truth, args.predictions)

    # Print results
    print_results(eval_results)

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(eval_results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
