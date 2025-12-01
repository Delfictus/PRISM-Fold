#!/usr/bin/env python3
"""
PRISM-LBS Validation Analysis Script v2.1
Perfectly aligned with PRISM-LBS v0.3.0 output format

Verified output structure (from test_protein.json):
{
  "structure": "...",
  "pockets": [
    {
      "atom_indices": [...],
      "residue_indices": [...],
      "centroid": [x, y, z],
      "volume": float,
      "enclosure_ratio": float,
      "mean_hydrophobicity": float,
      "mean_sasa": float,
      "mean_depth": float,
      "mean_flexibility": float,
      "mean_conservation": float,
      "persistence_score": float,
      "hbond_donors": int,
      "hbond_acceptors": int,
      "druggability_score": {
        "total": float,
        "classification": "Druggable" | "DifficultTarget",
        "components": {...}
      },
      "boundary_atoms": [...],
      "mean_electrostatic": float,
      "gnn_embedding": [],
      "gnn_druggability": float
    }
  ]
}
"""

import json
import csv
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict
import statistics


@dataclass
class PocketData:
    """Represents a single pocket from PRISM-LBS output."""
    pocket_id: int
    residue_indices: List[int]
    atom_indices: List[int]
    volume: float
    enclosure_ratio: float
    mean_hydrophobicity: float
    mean_sasa: float
    mean_depth: float
    mean_flexibility: float
    mean_conservation: float
    persistence_score: float
    hbond_donors: int
    hbond_acceptors: int
    druggability_total: float
    druggability_class: str
    centroid: List[float] = field(default_factory=list)
    
    @classmethod
    def from_json(cls, data: dict, pocket_id: int) -> 'PocketData':
        """Parse pocket from PRISM-LBS JSON output."""
        drug_score = data.get('druggability_score', {})
        return cls(
            pocket_id=pocket_id,
            residue_indices=data.get('residue_indices', []),
            atom_indices=data.get('atom_indices', []),
            volume=data.get('volume', 0.0),
            enclosure_ratio=data.get('enclosure_ratio', 0.0),
            mean_hydrophobicity=data.get('mean_hydrophobicity', 0.0),
            mean_sasa=data.get('mean_sasa', 0.0),
            mean_depth=data.get('mean_depth', 0.0),
            mean_flexibility=data.get('mean_flexibility', 0.0),
            mean_conservation=data.get('mean_conservation', 0.0),
            persistence_score=data.get('persistence_score', 0.0),
            hbond_donors=data.get('hbond_donors', 0),
            hbond_acceptors=data.get('hbond_acceptors', 0),
            druggability_total=drug_score.get('total', 0.0),
            druggability_class=drug_score.get('classification', 'Unknown'),
            centroid=data.get('centroid', [0, 0, 0])
        )


@dataclass
class StructureResult:
    """Results for a single structure."""
    pdb_id: str
    structure_name: str
    pockets: List[PocketData]
    
    @classmethod
    def from_json_file(cls, filepath: Path) -> Optional['StructureResult']:
        """Load structure results from PRISM-LBS JSON output."""
        try:
            with open(filepath) as f:
                data = json.load(f)
            
            pdb_id = filepath.stem.upper()
            structure_name = data.get('structure', pdb_id)
            
            pockets = []
            for i, pocket_data in enumerate(data.get('pockets', [])):
                pockets.append(PocketData.from_json(pocket_data, i + 1))
            
            return cls(pdb_id=pdb_id, structure_name=structure_name, pockets=pockets)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    @property
    def pocket_count(self) -> int:
        return len(self.pockets)
    
    @property
    def all_residue_indices(self) -> Set[int]:
        """Get unique residue indices across all pockets."""
        residues = set()
        for pocket in self.pockets:
            residues.update(pocket.residue_indices)
        return residues
    
    @property
    def top_pocket(self) -> Optional[PocketData]:
        """Get top pocket by druggability."""
        if not self.pockets:
            return None
        return max(self.pockets, key=lambda p: p.druggability_total)
    
    def get_high_flexibility_pockets(self, threshold: float = 25.0) -> List[PocketData]:
        """Get pockets with high flexibility (potential cryptic sites)."""
        return [p for p in self.pockets if p.mean_flexibility > threshold]
    
    def get_deep_pockets(self, threshold: float = 10.0) -> List[PocketData]:
        """Get deeply buried pockets."""
        return [p for p in self.pockets if p.mean_depth > threshold]
    
    def get_enclosed_pockets(self, threshold: float = 0.5) -> List[PocketData]:
        """Get well-enclosed pockets (good binding sites)."""
        return [p for p in self.pockets if p.enclosure_ratio > threshold]
    
    def get_druggable_pockets(self) -> List[PocketData]:
        """Get pockets classified as druggable."""
        return [p for p in self.pockets if p.druggability_class == 'Druggable']


def calculate_overlap(detected_residues: Set[int], ground_truth: str) -> float:
    """Calculate residue overlap between detection and ground truth."""
    if not ground_truth:
        return 0.0
    
    gt_residues = set()
    for res in ground_truth.split(';'):
        try:
            gt_residues.add(int(res.strip()))
        except ValueError:
            continue
    
    if not gt_residues:
        return 0.0
    
    overlap = len(detected_residues & gt_residues)
    return overlap / len(gt_residues)


def calculate_best_pocket_overlap(result: StructureResult, ground_truth: str) -> Tuple[float, int]:
    """Find the pocket with best overlap to ground truth."""
    best_overlap = 0.0
    best_pocket_id = 0
    
    for pocket in result.pockets[:10]:  # Top 10 pockets
        pocket_residues = set(pocket.residue_indices)
        overlap = calculate_overlap(pocket_residues, ground_truth)
        if overlap > best_overlap:
            best_overlap = overlap
            best_pocket_id = pocket.pocket_id
    
    return best_overlap, best_pocket_id


def load_ground_truth(filepath: Path) -> Dict[str, Dict]:
    """Load ground truth CSV file."""
    ground_truth = {}
    try:
        with open(filepath, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pdb_id = row.get('pdb_id', '').upper()
                if pdb_id:
                    ground_truth[pdb_id] = row
    except Exception as e:
        print(f"Error loading ground truth {filepath}: {e}")
    return ground_truth


class ValidationAnalyzer:
    """Analyzer for PRISM-LBS validation results."""
    
    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.tier_results: Dict[str, Dict[str, StructureResult]] = {}
        self.ground_truth: Dict[str, Dict[str, Dict]] = {}
        
    def load_results(self):
        """Load all tier results."""
        tier_dirs = {
            'tier1': 'tier1',
            'tier2_cryptosite': 'tier2_cryptosite', 
            'tier2_asbench': 'tier2_asbench',
            'tier3_novel': 'tier3_novel'
        }
        
        for tier_name, tier_dir in tier_dirs.items():
            tier_path = self.results_dir / tier_dir
            if not tier_path.exists():
                continue
                
            self.tier_results[tier_name] = {}
            for json_file in tier_path.glob('*.json'):
                result = StructureResult.from_json_file(json_file)
                if result:
                    self.tier_results[tier_name][result.pdb_id] = result
    
    def load_ground_truth(self, ground_truth_dir: Path):
        """Load all ground truth files."""
        gt_files = {
            'tier1': 'tier1_binding_sites.csv',
            'tier2_cryptosite': 'cryptosite_ground_truth.csv',
            'tier2_asbench': 'asbench_ground_truth.csv',
            'tier3_novel': 'novel_targets_ground_truth.csv'
        }
        
        for tier_name, filename in gt_files.items():
            filepath = ground_truth_dir / filename
            if filepath.exists():
                self.ground_truth[tier_name] = load_ground_truth(filepath)
    
    def analyze_tier(self, tier_name: str, residue_field: str, threshold: float) -> Dict:
        """Analyze results for a single tier."""
        results = self.tier_results.get(tier_name, {})
        gt = self.ground_truth.get(tier_name, {})
        
        passed = 0
        total = 0
        details = []
        
        for pdb_id, result in results.items():
            pdb_gt = gt.get(pdb_id.upper()) or gt.get(pdb_id.lower())
            if not pdb_gt:
                continue
            
            gt_residues = pdb_gt.get(residue_field, '')
            overlap, best_pocket = calculate_best_pocket_overlap(result, gt_residues)
            
            status = 'PASS' if overlap >= threshold else 'FAIL'
            if status == 'PASS':
                passed += 1
            total += 1
            
            details.append({
                'pdb_id': pdb_id,
                'pocket_count': result.pocket_count,
                'best_pocket': best_pocket,
                'overlap': overlap,
                'top_druggability': result.top_pocket.druggability_total if result.top_pocket else 0,
                'status': status
            })
        
        rate = (passed / total * 100) if total > 0 else 0
        
        return {
            'tier': tier_name,
            'passed': passed,
            'total': total,
            'rate': rate,
            'threshold': threshold * 100,
            'details': details
        }
    
    def analyze_all(self) -> Dict:
        """Analyze all tiers."""
        return {
            'tier1': self.analyze_tier('tier1', 'binding_residues', 0.40),
            'tier2_cryptosite': self.analyze_tier('tier2_cryptosite', 'cryptic_residues', 0.35),
            'tier2_asbench': self.analyze_tier('tier2_asbench', 'allosteric_residues', 0.30),
            'tier3_novel': self.analyze_tier('tier3_novel', 'site_residues', 0.30)
        }
    
    def get_detection_statistics(self) -> Dict:
        """Get overall detection statistics."""
        all_pockets = []
        for tier_results in self.tier_results.values():
            for result in tier_results.values():
                all_pockets.extend(result.pockets)
        
        if not all_pockets:
            return {}
        
        volumes = [p.volume for p in all_pockets if p.volume > 0]
        druggabilities = [p.druggability_total for p in all_pockets]
        flexibilities = [p.mean_flexibility for p in all_pockets]
        depths = [p.mean_depth for p in all_pockets]
        
        return {
            'total_structures': sum(len(r) for r in self.tier_results.values()),
            'total_pockets': len(all_pockets),
            'volume': {
                'mean': statistics.mean(volumes) if volumes else 0,
                'median': statistics.median(volumes) if volumes else 0,
                'min': min(volumes) if volumes else 0,
                'max': max(volumes) if volumes else 0
            },
            'druggability': {
                'mean': statistics.mean(druggabilities),
                'druggable_count': sum(1 for p in all_pockets if p.druggability_class == 'Druggable')
            },
            'flexibility': {
                'mean': statistics.mean(flexibilities),
                'high_flex_count': sum(1 for p in all_pockets if p.mean_flexibility > 25)
            },
            'depth': {
                'mean': statistics.mean(depths),
                'deep_count': sum(1 for p in all_pockets if p.mean_depth > 10)
            }
        }
    
    def export_detailed_csv(self, output_path: Path):
        """Export detailed results to CSV."""
        fieldnames = [
            'tier', 'pdb_id', 'pocket_id', 'residue_count', 'atom_count',
            'volume', 'enclosure_ratio', 'mean_depth', 'mean_flexibility',
            'mean_hydrophobicity', 'mean_sasa', 'mean_conservation',
            'hbond_donors', 'hbond_acceptors',
            'druggability_total', 'druggability_class'
        ]
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for tier_name, tier_results in self.tier_results.items():
                for pdb_id, result in tier_results.items():
                    for pocket in result.pockets:
                        writer.writerow({
                            'tier': tier_name,
                            'pdb_id': pdb_id,
                            'pocket_id': pocket.pocket_id,
                            'residue_count': len(pocket.residue_indices),
                            'atom_count': len(pocket.atom_indices),
                            'volume': f"{pocket.volume:.2f}",
                            'enclosure_ratio': f"{pocket.enclosure_ratio:.3f}",
                            'mean_depth': f"{pocket.mean_depth:.2f}",
                            'mean_flexibility': f"{pocket.mean_flexibility:.2f}",
                            'mean_hydrophobicity': f"{pocket.mean_hydrophobicity:.2f}",
                            'mean_sasa': f"{pocket.mean_sasa:.2f}",
                            'mean_conservation': f"{pocket.mean_conservation:.3f}",
                            'hbond_donors': pocket.hbond_donors,
                            'hbond_acceptors': pocket.hbond_acceptors,
                            'druggability_total': f"{pocket.druggability_total:.4f}",
                            'druggability_class': pocket.druggability_class
                        })
    
    def print_summary(self, analysis: Dict):
        """Print formatted summary."""
        print("\n" + "="*80)
        print("PRISM-LBS VALIDATION ANALYSIS SUMMARY")
        print("="*80 + "\n")
        
        # Tier results
        print("TIER RESULTS:")
        print("-"*60)
        
        tier_names = {
            'tier1': 'Tier 1 (Table Stakes)',
            'tier2_cryptosite': 'Tier 2A (CryptoSite)',
            'tier2_asbench': 'Tier 2B (ASBench)',
            'tier3_novel': 'Tier 3 (Novel Targets)'
        }
        
        targets = {
            'tier1': 85,
            'tier2_cryptosite': 65,
            'tier2_asbench': 65,
            'tier3_novel': 50
        }
        
        all_passed = True
        for tier_key, tier_display in tier_names.items():
            tier_data = analysis.get(tier_key, {})
            rate = tier_data.get('rate', 0)
            passed = tier_data.get('passed', 0)
            total = tier_data.get('total', 0)
            target = targets[tier_key]
            
            status = "✅ PASS" if rate >= target else "❌ FAIL"
            if rate < target:
                all_passed = False
            
            print(f"  {tier_display:30} {passed:2}/{total:2} ({rate:5.1f}%) Target: {target}%  {status}")
        
        print("-"*60)
        
        # Statistics
        stats = self.get_detection_statistics()
        if stats:
            print(f"\nOVERALL STATISTICS:")
            print(f"  Total structures analyzed: {stats['total_structures']}")
            print(f"  Total pockets detected: {stats['total_pockets']}")
            print(f"  Average pocket volume: {stats['volume']['mean']:.1f} Å³")
            print(f"  Average druggability: {stats['druggability']['mean']:.3f}")
            print(f"  Druggable pockets: {stats['druggability']['druggable_count']}")
            print(f"  High-flexibility pockets: {stats['flexibility']['high_flex_count']}")
        
        # Publication readiness
        print("\nPUBLICATION READINESS:")
        print("-"*60)
        
        tier1_pass = analysis.get('tier1', {}).get('rate', 0) >= 85
        tier2a_pass = analysis.get('tier2_cryptosite', {}).get('rate', 0) >= 65
        tier2b_pass = analysis.get('tier2_asbench', {}).get('rate', 0) >= 65
        
        tier2a_rate = analysis.get('tier2_cryptosite', {}).get('rate', 0)
        tier2b_rate = analysis.get('tier2_asbench', {}).get('rate', 0)
        
        if all_passed:
            print("  🎉 ALL TIERS PASSED - Ready for Bioinformatics or NAR")
            if tier2a_rate >= 80 and tier2b_rate >= 80:
                print("  🏆 EXCEPTIONAL - Consider Nature Communications")
        elif tier1_pass and tier2a_pass:
            print("  📝 Ready for J. Chem. Inf. Model.")
        else:
            print("  🔧 More optimization needed")
        
        print("\n" + "="*80)


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_benchmark.py <results_directory>")
        print("       python analyze_benchmark.py benchmark/complete_validation")
        sys.exit(1)
    
    results_dir = Path(sys.argv[1])
    
    if not results_dir.exists():
        print(f"Error: Directory {results_dir} does not exist")
        sys.exit(1)
    
    # Determine paths
    if (results_dir / 'results').exists():
        actual_results = results_dir / 'results'
        gt_dir = results_dir / 'ground_truth'
    else:
        actual_results = results_dir
        gt_dir = results_dir.parent / 'ground_truth'
    
    # Initialize analyzer
    analyzer = ValidationAnalyzer(actual_results)
    
    # Load data
    print("Loading results...")
    analyzer.load_results()
    
    print("Loading ground truth...")
    analyzer.load_ground_truth(gt_dir)
    
    # Analyze
    print("Analyzing...")
    analysis = analyzer.analyze_all()
    
    # Export
    output_dir = actual_results / 'analysis'
    output_dir.mkdir(exist_ok=True)
    
    csv_path = output_dir / 'detailed_analysis.csv'
    analyzer.export_detailed_csv(csv_path)
    print(f"Exported detailed results to: {csv_path}")
    
    json_path = output_dir / 'analysis_summary.json'
    with open(json_path, 'w') as f:
        json.dump({
            'analysis': analysis,
            'statistics': analyzer.get_detection_statistics()
        }, f, indent=2)
    print(f"Exported summary to: {json_path}")
    
    # Print summary
    analyzer.print_summary(analysis)


if __name__ == '__main__':
    main()
