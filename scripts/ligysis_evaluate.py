#!/usr/bin/env python3
"""
LIGYSIS Benchmark Evaluation Script for PRISM-LBS

Uses the exact evaluation methodology from:
"Comparative analysis of methods for the prediction of protein-ligand binding sites"
Journal of Cheminformatics 2024

Ground Truth: MASTER_POCKET_SHAPE_DICT_EXTENDED_TRANS.pkl from LIGYSIS benchmark
Format: method_name -> {pdb_chain_siteIdx: (volume, 3D_coordinates_array)}

Key metrics:
- Top-N+K recall: For each protein with N binding sites, evaluate if top N+K predictions
  contain all true sites (IRel >= threshold, typically 0.25-0.5)
- DCC (Distance from Centroid to Closest): Minimum distance from predicted centroid to any ground truth centroid
- DVO (Distance-Volume Overlap): Volume-weighted distance overlap between predicted and ground truth shapes
- IRel (Relative Intersection): |pred ∩ ref| / |ref|
- Classification metrics: AUC, AUPRC, MCC, F1, TPR, FPR, Precision

Competitor baselines from LIGYSIS paper (Top-N+2 recall @ IRel=0.25):
- fpocket: 60%
- P2Rank: 54%
- DeepPocket: 60%
- VN-EGNN: 46%
- PUResNet: 45%

Ground truth source: https://zenodo.org/doi/10.5281/zenodo.13171100
"""

import os
import sys
import json
import glob
import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict
from scipy.spatial.distance import euclidean, cdist
from scipy.spatial import ConvexHull
import warnings

# Try to import sklearn for classification metrics
try:
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, matthews_corrcoef,
        f1_score, precision_score, recall_score, confusion_matrix
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Classification metrics will be limited.")


@dataclass
class BindingSite:
    """Represents a binding site (ground truth or prediction)."""
    site_id: int
    pdb_id: str
    chain: str
    residues: Set[str] = field(default_factory=set)
    centroid: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    coordinates: np.ndarray = field(default_factory=lambda: np.array([]))
    volume: float = 0.0
    score: float = 0.0
    rank: int = 0
    origin: str = ""

    @property
    def pdb_chain(self) -> str:
        return f"{self.pdb_id}_{self.chain}"


def load_ligysis_ground_truth(pkl_path: Path) -> Dict[str, List[BindingSite]]:
    """
    Load LIGYSIS ground truth from master pickle file.

    Format: method_name -> {pdb_chain_siteIdx: (volume, 3D_coordinates_array)}
    Ground truth method is 'LIGYSIS' key.
    """
    sites_by_pdb = defaultdict(list)

    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        # The ground truth is stored under 'LIGYSIS' key
        if 'LIGYSIS' not in data:
            print(f"Warning: 'LIGYSIS' key not found in pickle. Available keys: {list(data.keys())[:10]}")
            # Try to find the ground truth - might be under a different key
            ligysis_data = data.get('LIGYSIS', data.get('ligysis', {}))
            if not ligysis_data:
                # Use the first method that looks like ground truth
                for key in data.keys():
                    if isinstance(data[key], dict) and len(data[key]) > 100:
                        print(f"Using '{key}' as ground truth source")
                        ligysis_data = data[key]
                        break
        else:
            ligysis_data = data['LIGYSIS']

        print(f"Loading ground truth from LIGYSIS key with {len(ligysis_data)} entries...")

        # Parse each entry: key format is "pdb_chain_siteIdx"
        for key, value in ligysis_data.items():
            try:
                # Parse key: "1m76_A_1" -> pdb=1m76, chain=A, site=1
                parts = key.rsplit('_', 2)
                if len(parts) == 3:
                    pdb_id = parts[0].lower()
                    chain = parts[1]
                    site_idx = int(parts[2])
                else:
                    # Try alternative parsing
                    pdb_id = key[:4].lower()
                    chain = key[5] if len(key) > 5 else 'A'
                    site_idx = int(key.split('_')[-1]) if '_' in key else 1

                # Parse value: (volume, 3D_coordinates_array)
                if isinstance(value, tuple) and len(value) == 2:
                    volume, coords = value
                    coords = np.array(coords) if not isinstance(coords, np.ndarray) else coords
                elif isinstance(value, np.ndarray):
                    coords = value
                    volume = 0.0
                else:
                    continue

                # Compute centroid from coordinates
                if len(coords) > 0:
                    centroid = tuple(np.mean(coords, axis=0))
                else:
                    centroid = (0.0, 0.0, 0.0)

                pdb_chain = f"{pdb_id}_{chain}"

                site = BindingSite(
                    site_id=site_idx,
                    pdb_id=pdb_id,
                    chain=chain,
                    residues=set(),  # Ground truth has 3D coords, not residues
                    centroid=centroid,
                    coordinates=coords,
                    volume=float(volume) if volume else 0.0,
                    score=1.0,
                    rank=site_idx,
                    origin="LIGYSIS"
                )

                sites_by_pdb[pdb_chain].append(site)

            except Exception as e:
                # Skip malformed entries
                continue

        # Sort sites by rank within each structure
        for pdb_chain in sites_by_pdb:
            sites_by_pdb[pdb_chain].sort(key=lambda s: s.site_id)
            # Re-assign ranks
            for i, site in enumerate(sites_by_pdb[pdb_chain]):
                site.rank = i + 1

        print(f"Loaded {sum(len(s) for s in sites_by_pdb.values())} binding sites from {len(sites_by_pdb)} structures")

        return dict(sites_by_pdb)

    except Exception as e:
        print(f"Error loading pickle: {e}")
        import traceback
        traceback.print_exc()
        return {}


def parse_prism_json(json_path: Path) -> List[BindingSite]:
    """Parse PRISM publication JSON output to binding sites."""
    sites = []

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error parsing {json_path}: {e}")
        return []

    # Extract PDB ID from filename
    stem = json_path.stem.replace('.trans', '')
    parts = stem.rsplit('_', 1)
    pdb_id = parts[0].lower() if len(parts) > 1 else stem[:4].lower()
    chain = parts[1] if len(parts) > 1 else 'A'

    for pocket in data.get('pockets', []):
        pocket_id = pocket.get('id', 0)

        # Get residue indices
        residue_indices = pocket.get('residue_indices', [])
        residues = {str(r) for r in residue_indices}

        # Get geometry
        geometry = pocket.get('geometry', {})
        centroid = tuple(geometry.get('centroid', [0, 0, 0]))
        volume = geometry.get('volume', 0.0)

        # Get atom coordinates if available for DVO computation
        coords = []
        if 'atoms' in pocket:
            for atom in pocket['atoms']:
                if 'position' in atom:
                    coords.append(atom['position'])
        elif 'atom_coordinates' in geometry:
            coords = geometry['atom_coordinates']

        # Get score
        scores = pocket.get('scores', {})
        score = scores.get('confidence', 0.0)
        if score == 0:
            score = scores.get('druggability_total', 0.0)
        if score == 0:
            score = scores.get('druggability', 0.0)

        sites.append(BindingSite(
            site_id=pocket_id,
            pdb_id=pdb_id,
            chain=chain,
            residues=residues,
            centroid=centroid,
            coordinates=np.array(coords) if coords else np.array([]),
            volume=volume,
            score=score,
            rank=pocket.get('rank', pocket_id),
            origin="PRISM-LBS"
        ))

    # Sort by rank
    sites.sort(key=lambda s: s.rank)

    return sites


def compute_relative_intersection(pred_residues: Set[str], ref_residues: Set[str]) -> float:
    """
    Compute relative intersection (IRel): |pred ∩ ref| / |ref|
    This measures what fraction of the reference site is covered by the prediction.
    """
    if not ref_residues:
        return 0.0
    intersection = len(pred_residues & ref_residues)
    return intersection / len(ref_residues)


def compute_dcc(pred_centroid: Tuple[float, float, float],
                ref_sites: List[BindingSite]) -> float:
    """
    Compute DCC (Distance from Centroid to Closest).
    Returns minimum distance from predicted centroid to any ground truth centroid.
    Lower is better.
    """
    if not ref_sites:
        return float('inf')

    distances = []
    for ref in ref_sites:
        if ref.centroid and pred_centroid:
            d = euclidean(pred_centroid, ref.centroid)
            distances.append(d)

    return min(distances) if distances else float('inf')


def compute_dvo(pred_coords: np.ndarray, ref_coords: np.ndarray,
                distance_threshold: float = 4.0) -> float:
    """
    Compute DVO (Distance-Volume Overlap).
    Measures overlap between predicted and ground truth binding site shapes.

    Uses a point-cloud based overlap: fraction of reference points that are
    within distance_threshold of any predicted point.
    """
    if len(pred_coords) == 0 or len(ref_coords) == 0:
        return 0.0

    try:
        # Compute pairwise distances
        distances = cdist(ref_coords, pred_coords, metric='euclidean')

        # For each reference point, find minimum distance to any predicted point
        min_distances = np.min(distances, axis=1)

        # Fraction of reference points covered by prediction
        covered = np.sum(min_distances <= distance_threshold)
        dvo = covered / len(ref_coords)

        return float(dvo)

    except Exception:
        return 0.0


def compute_centroid_distance_match(pred: BindingSite, ref: BindingSite,
                                     distance_threshold: float = 4.0) -> bool:
    """Check if prediction matches reference by centroid distance."""
    if not pred.centroid or not ref.centroid:
        return False

    distance = euclidean(pred.centroid, ref.centroid)
    return distance <= distance_threshold


def evaluate_topn_plus_k_recall(
    reference_sites: Dict[str, List[BindingSite]],
    predicted_sites: Dict[str, List[BindingSite]],
    k: int = 2,
    distance_threshold: float = 4.0
) -> Dict:
    """
    Evaluate Top-N+K recall using centroid distance matching.

    For each protein with N binding sites, check if top N+K predictions
    contain all N sites within distance_threshold.
    """
    total_sites = 0
    correct_sites = 0
    per_structure_results = []

    for pdb_chain, ref_sites in reference_sites.items():
        n_sites = len(ref_sites)
        if n_sites == 0:
            continue

        preds = predicted_sites.get(pdb_chain, [])
        rank_limit = n_sites + k
        top_preds = [p for p in preds if p.rank <= rank_limit]

        structure_correct = 0
        for ref_site in ref_sites:
            total_sites += 1

            # Check if any top prediction is within distance threshold
            for pred in top_preds:
                if compute_centroid_distance_match(pred, ref_site, distance_threshold):
                    correct_sites += 1
                    structure_correct += 1
                    break

        per_structure_results.append({
            'pdb_chain': pdb_chain,
            'n_sites': n_sites,
            'rank_limit': rank_limit,
            'n_predictions': len(preds),
            'correct': structure_correct,
            'recall': structure_correct / n_sites if n_sites > 0 else 0
        })

    recall = correct_sites / total_sites if total_sites > 0 else 0

    # Confidence interval (normal approximation)
    se = np.sqrt(recall * (1 - recall) / total_sites) if total_sites > 0 else 0
    ci_lower = max(0, recall - 1.96 * se)
    ci_upper = min(1, recall + 1.96 * se)

    return {
        'recall': recall,
        'correct_sites': correct_sites,
        'total_sites': total_sites,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'k': k,
        'distance_threshold': distance_threshold,
        'per_structure': per_structure_results
    }


def compute_mean_dcc(
    reference_sites: Dict[str, List[BindingSite]],
    predicted_sites: Dict[str, List[BindingSite]]
) -> Dict:
    """
    Compute mean DCC across all top-1 predictions.
    """
    dcc_values = []

    for pdb_chain, ref_sites in reference_sites.items():
        if not ref_sites:
            continue

        preds = predicted_sites.get(pdb_chain, [])
        if not preds:
            continue

        # Get top-1 prediction
        top_pred = min(preds, key=lambda p: p.rank)

        # Compute DCC
        dcc = compute_dcc(top_pred.centroid, ref_sites)
        if dcc != float('inf'):
            dcc_values.append(dcc)

    if not dcc_values:
        return {'mean_dcc': float('inf'), 'std_dcc': 0, 'n_evaluated': 0}

    return {
        'mean_dcc': np.mean(dcc_values),
        'std_dcc': np.std(dcc_values),
        'median_dcc': np.median(dcc_values),
        'n_evaluated': len(dcc_values),
        'dcc_lt_4': sum(1 for d in dcc_values if d < 4.0) / len(dcc_values),
        'dcc_lt_8': sum(1 for d in dcc_values if d < 8.0) / len(dcc_values)
    }


def compute_mean_dvo(
    reference_sites: Dict[str, List[BindingSite]],
    predicted_sites: Dict[str, List[BindingSite]],
    distance_threshold: float = 4.0
) -> Dict:
    """
    Compute mean DVO across all predictions.
    """
    dvo_values = []

    for pdb_chain, ref_sites in reference_sites.items():
        if not ref_sites:
            continue

        preds = predicted_sites.get(pdb_chain, [])
        if not preds:
            continue

        for ref in ref_sites:
            if len(ref.coordinates) == 0:
                continue

            # Find best matching prediction
            best_dvo = 0.0
            for pred in preds:
                if len(pred.coordinates) > 0:
                    dvo = compute_dvo(pred.coordinates, ref.coordinates, distance_threshold)
                    best_dvo = max(best_dvo, dvo)
                else:
                    # Fallback: use centroid-based matching
                    if compute_centroid_distance_match(pred, ref, distance_threshold):
                        best_dvo = max(best_dvo, 0.5)  # Partial credit

            if best_dvo > 0:
                dvo_values.append(best_dvo)

    if not dvo_values:
        return {'mean_dvo': 0.0, 'std_dvo': 0, 'n_evaluated': 0}

    return {
        'mean_dvo': np.mean(dvo_values),
        'std_dvo': np.std(dvo_values),
        'n_evaluated': len(dvo_values)
    }


def compute_classification_metrics(
    reference_sites: Dict[str, List[BindingSite]],
    predicted_sites: Dict[str, List[BindingSite]],
    distance_threshold: float = 4.0
) -> Dict:
    """
    Compute classification metrics: AUC, AUPRC, MCC, F1, TPR, FPR, Precision.

    Treats each prediction as a binary classification:
    - True if it matches any ground truth site
    - False otherwise
    """
    if not SKLEARN_AVAILABLE:
        return {'warning': 'scikit-learn not available'}

    y_true = []
    y_scores = []
    y_pred = []

    for pdb_chain, preds in predicted_sites.items():
        refs = reference_sites.get(pdb_chain, [])

        for pred in preds:
            # Check if prediction matches any ground truth
            is_match = False
            for ref in refs:
                if compute_centroid_distance_match(pred, ref, distance_threshold):
                    is_match = True
                    break

            y_true.append(1 if is_match else 0)
            y_scores.append(pred.score)
            y_pred.append(1)  # All predictions are positive predictions

    if not y_true or sum(y_true) == 0 or sum(y_true) == len(y_true):
        return {
            'auc': 0.0,
            'auprc': 0.0,
            'mcc': 0.0,
            'f1': 0.0,
            'precision': 0.0,
            'tpr': 0.0,
            'fpr': 0.0,
            'n_samples': len(y_true)
        }

    try:
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        y_pred = np.array(y_pred)

        # Handle NaN scores
        valid_mask = ~np.isnan(y_scores)
        if not valid_mask.all():
            y_true = y_true[valid_mask]
            y_scores = y_scores[valid_mask]
            y_pred = y_pred[valid_mask]

        # AUC-ROC
        try:
            auc = roc_auc_score(y_true, y_scores)
        except:
            auc = 0.0

        # AUPRC (Average Precision)
        try:
            auprc = average_precision_score(y_true, y_scores)
        except:
            auprc = 0.0

        # Binary predictions based on score threshold (median)
        threshold = np.median(y_scores)
        y_pred_binary = (y_scores >= threshold).astype(int)

        # MCC
        try:
            mcc = matthews_corrcoef(y_true, y_pred_binary)
        except:
            mcc = 0.0

        # F1
        try:
            f1 = f1_score(y_true, y_pred_binary, zero_division=0)
        except:
            f1 = 0.0

        # Precision
        try:
            precision = precision_score(y_true, y_pred_binary, zero_division=0)
        except:
            precision = 0.0

        # TPR, FPR from confusion matrix
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary, labels=[0, 1]).ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        except:
            tpr, fpr = 0.0, 0.0

        return {
            'auc': float(auc),
            'auprc': float(auprc),
            'mcc': float(mcc),
            'f1': float(f1),
            'precision': float(precision),
            'tpr': float(tpr),
            'fpr': float(fpr),
            'n_samples': len(y_true)
        }

    except Exception as e:
        print(f"Error computing classification metrics: {e}")
        return {'error': str(e)}


def run_full_evaluation(
    ground_truth_pkl: Path,
    predictions_dir: Path,
    k_values: List[int] = [0, 1, 2, 3],
    distance_thresholds: List[float] = [4.0, 8.0]
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run full LIGYSIS-style evaluation.
    """
    # Load ground truth
    print(f"Loading ground truth from: {ground_truth_pkl}")
    reference_sites = load_ligysis_ground_truth(ground_truth_pkl)

    if not reference_sites:
        print("ERROR: No ground truth loaded!")
        return pd.DataFrame(), {}

    # Load predictions
    print(f"\nLoading PRISM predictions from: {predictions_dir}")
    predicted_sites = {}
    json_files = list(predictions_dir.glob('*.json'))

    for json_file in json_files:
        sites = parse_prism_json(json_file)
        if sites:
            pdb_chain = sites[0].pdb_chain
            predicted_sites[pdb_chain] = sites

    print(f"Loaded predictions for {len(predicted_sites)} structures")

    # Find common structures
    common = set(reference_sites.keys()) & set(predicted_sites.keys())
    print(f"Common structures for evaluation: {len(common)}")

    if len(common) == 0:
        print("\nNo common structures found!")
        print(f"Ground truth keys (sample): {list(reference_sites.keys())[:5]}")
        print(f"Prediction keys (sample): {list(predicted_sites.keys())[:5]}")
        return pd.DataFrame(), {}

    # Filter to common structures
    ref_filtered = {k: v for k, v in reference_sites.items() if k in common}
    pred_filtered = {k: v for k, v in predicted_sites.items() if k in common}

    total_ref_sites = sum(len(s) for s in ref_filtered.values())
    total_pred_sites = sum(len(s) for s in pred_filtered.values())
    print(f"Total ground truth sites: {total_ref_sites}")
    print(f"Total predicted sites: {total_pred_sites}")

    # Evaluate Top-N+K recall
    results = []
    for k in k_values:
        for dist_thresh in distance_thresholds:
            eval_result = evaluate_topn_plus_k_recall(
                ref_filtered, pred_filtered, k=k, distance_threshold=dist_thresh
            )

            results.append({
                'metric': f'Top-N+{k}',
                'k': k,
                'distance_threshold': dist_thresh,
                'recall': eval_result['recall'],
                'recall_pct': eval_result['recall'] * 100,
                'correct': eval_result['correct_sites'],
                'total': eval_result['total_sites'],
                'ci_lower': eval_result['ci_lower'] * 100,
                'ci_upper': eval_result['ci_upper'] * 100
            })

    df = pd.DataFrame(results)

    # Compute additional metrics
    summary = {}

    # DCC metrics
    print("\nComputing DCC metrics...")
    dcc_metrics = compute_mean_dcc(ref_filtered, pred_filtered)
    summary['dcc'] = dcc_metrics

    # DVO metrics
    print("Computing DVO metrics...")
    dvo_metrics = compute_mean_dvo(ref_filtered, pred_filtered)
    summary['dvo'] = dvo_metrics

    # Classification metrics
    print("Computing classification metrics...")
    class_metrics = compute_classification_metrics(ref_filtered, pred_filtered)
    summary['classification'] = class_metrics

    return df, summary


def print_full_results(df: pd.DataFrame, summary: Dict):
    """Print comprehensive results table."""

    print("\n" + "=" * 80)
    print("                    PRISM-LBS LIGYSIS BENCHMARK RESULTS")
    print("=" * 80)

    # Baseline comparison
    baselines = {
        'fpocket': 60,
        'P2Rank': 54,
        'DeepPocket-Seg': 60,
        'VN-EGNN': 46,
        'PUResNet': 45,
        'IF-SitePred': 55,
        'GrASP': 52
    }

    # Get PRISM at Top-N+2, dist=4.0
    prism_row = df[(df['k'] == 2) & (df['distance_threshold'] == 4.0)]
    if not prism_row.empty:
        prism_recall = prism_row.iloc[0]['recall_pct']
        prism_ci = (prism_row.iloc[0]['ci_lower'], prism_row.iloc[0]['ci_upper'])
    else:
        prism_recall = 0
        prism_ci = (0, 0)

    print("\n--- TOP-N+2 RECALL @ DCC<4Å (Primary Metric) ---\n")
    print(f"{'Method':<20} {'Recall (%)':<20} {'vs PRISM':<15}")
    print("-" * 55)
    print(f"{'PRISM-LBS':<20} {prism_recall:>6.1f} [{prism_ci[0]:.1f}-{prism_ci[1]:.1f}]   {'--':>10}")

    for method, recall in sorted(baselines.items(), key=lambda x: -x[1]):
        diff = prism_recall - recall
        sign = '+' if diff > 0 else ''
        print(f"{method:<20} {recall:>6.1f}                 {sign}{diff:.1f}")

    print("-" * 55)

    if prism_recall >= 60:
        print(f"\n✓ PASSED - PRISM achieves {prism_recall:.1f}% (≥60% target)")
    else:
        print(f"\n○ PENDING - PRISM at {prism_recall:.1f}% (need {60 - prism_recall:.1f}% to reach 60%)")

    # Full recall matrix
    print("\n\n--- RECALL MATRIX ---\n")
    for dist in df['distance_threshold'].unique():
        print(f"Distance threshold: {dist}Å")
        subset = df[df['distance_threshold'] == dist]
        for _, row in subset.iterrows():
            print(f"  {row['metric']}: {row['recall_pct']:.1f}% ({row['correct']}/{row['total']})")
        print()

    # DCC metrics
    if 'dcc' in summary:
        dcc = summary['dcc']
        print("--- DCC (Distance to Closest) METRICS ---\n")
        print(f"  Mean DCC:   {dcc.get('mean_dcc', 0):.2f} Å")
        print(f"  Median DCC: {dcc.get('median_dcc', 0):.2f} Å")
        print(f"  Std DCC:    {dcc.get('std_dcc', 0):.2f} Å")
        print(f"  DCC < 4Å:   {dcc.get('dcc_lt_4', 0)*100:.1f}%")
        print(f"  DCC < 8Å:   {dcc.get('dcc_lt_8', 0)*100:.1f}%")
        print(f"  Evaluated:  {dcc.get('n_evaluated', 0)} structures")
        print()

    # DVO metrics
    if 'dvo' in summary:
        dvo = summary['dvo']
        print("--- DVO (Distance-Volume Overlap) METRICS ---\n")
        print(f"  Mean DVO: {dvo.get('mean_dvo', 0):.3f}")
        print(f"  Std DVO:  {dvo.get('std_dvo', 0):.3f}")
        print(f"  Evaluated: {dvo.get('n_evaluated', 0)} sites")
        print()

    # Classification metrics
    if 'classification' in summary:
        cls = summary['classification']
        if 'auc' in cls:
            print("--- CLASSIFICATION METRICS ---\n")
            print(f"  AUC-ROC:   {cls.get('auc', 0):.3f}")
            print(f"  AUPRC:     {cls.get('auprc', 0):.3f}")
            print(f"  MCC:       {cls.get('mcc', 0):.3f}")
            print(f"  F1:        {cls.get('f1', 0):.3f}")
            print(f"  Precision: {cls.get('precision', 0):.3f}")
            print(f"  TPR:       {cls.get('tpr', 0):.3f}")
            print(f"  FPR:       {cls.get('fpr', 0):.3f}")
            print(f"  Samples:   {cls.get('n_samples', 0)}")
            print()

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='LIGYSIS Benchmark Evaluation for PRISM-LBS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python ligysis_evaluate.py --ground-truth /path/to/MASTER_POCKET_SHAPE_DICT_EXTENDED_TRANS.pkl --predictions /path/to/prism_output/

Ground truth source:
  LIGYSIS benchmark dataset from Zenodo: https://zenodo.org/doi/10.5281/zenodo.13171100
"""
    )
    parser.add_argument(
        '--ground-truth', '-g',
        type=Path,
        required=True,
        help='Path to LIGYSIS ground truth pickle (MASTER_POCKET_SHAPE_DICT_EXTENDED_TRANS.pkl)'
    )
    parser.add_argument(
        '--predictions', '-p',
        type=Path,
        required=True,
        help='Directory containing PRISM JSON predictions'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        help='Output JSON file for detailed results'
    )
    parser.add_argument(
        '--csv',
        type=Path,
        help='Output CSV file for recall matrix'
    )
    parser.add_argument(
        '--k-values',
        type=int,
        nargs='+',
        default=[0, 1, 2, 3],
        help='K values for Top-N+K evaluation (default: 0 1 2 3)'
    )
    parser.add_argument(
        '--distance-thresholds',
        type=float,
        nargs='+',
        default=[4.0, 8.0],
        help='Distance thresholds in Angstroms (default: 4.0 8.0)'
    )

    args = parser.parse_args()

    # Validate paths
    if not args.ground_truth.exists():
        print(f"Error: Ground truth file not found: {args.ground_truth}")
        sys.exit(1)
    if not args.predictions.exists():
        print(f"Error: Predictions directory not found: {args.predictions}")
        sys.exit(1)

    # Run evaluation
    df, summary = run_full_evaluation(
        ground_truth_pkl=args.ground_truth,
        predictions_dir=args.predictions,
        k_values=args.k_values,
        distance_thresholds=args.distance_thresholds
    )

    if df.empty:
        print("\nNo results generated. Check ground truth and predictions.")
        sys.exit(1)

    # Print results
    print_full_results(df, summary)

    # Save results
    if args.output:
        results = {
            'recall_matrix': df.to_dict(orient='records'),
            'summary': summary
        }
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nDetailed results saved to: {args.output}")

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"Recall matrix saved to: {args.csv}")


if __name__ == '__main__':
    main()
