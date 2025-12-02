#!/usr/bin/env python3
"""
Download LIGYSIS benchmark structures from RCSB PDB.

The LIGYSIS ground truth uses original RCSB coordinates, not transformed ones.
This script downloads the original structures needed for evaluation.

Usage:
  python scripts/download_ligysis_structures.py [--quick] [--output DIR]

Options:
  --quick    Download only first 100 structures for testing
  --output   Output directory (default: benchmark/ligysis_structures)
"""

import os
import sys
import json
import pickle
import argparse
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

def download_structure(pdb_id: str, chain: str, output_dir: Path) -> tuple:
    """Download a structure from RCSB."""
    pdb_id = pdb_id.lower()

    # Try PDB first (better PRISM compatibility), then CIF as fallback
    for ext, url_template in [
        ('.pdb', f'https://files.rcsb.org/download/{pdb_id}.pdb'),
        ('.cif', f'https://files.rcsb.org/download/{pdb_id}.cif'),
    ]:
        output_file = output_dir / f'{pdb_id}_{chain}{ext}'
        if output_file.exists():
            return (pdb_id, chain, True, 'cached')

        try:
            urllib.request.urlretrieve(url_template, output_file)
            return (pdb_id, chain, True, ext)
        except Exception:
            continue

    return (pdb_id, chain, False, 'failed')


def main():
    parser = argparse.ArgumentParser(description='Download LIGYSIS structures from RCSB')
    parser.add_argument('--quick', action='store_true', help='Download only 100 structures')
    parser.add_argument('--output', type=Path, default=Path('benchmark/ligysis_structures'),
                        help='Output directory')
    parser.add_argument('--ground-truth', type=Path,
                        default=Path('benchmark/true_benchmarks/ligysis_true/MASTER_POCKET_SHAPE_DICT_EXTENDED_TRANS.pkl'),
                        help='Ground truth pickle file')
    parser.add_argument('--parallel', type=int, default=8, help='Parallel downloads')
    args = parser.parse_args()

    # Load ground truth to get list of structures
    if not args.ground_truth.exists():
        print(f"Error: Ground truth file not found: {args.ground_truth}")
        sys.exit(1)

    print(f"Loading ground truth from {args.ground_truth}...")
    with open(args.ground_truth, 'rb') as f:
        data = pickle.load(f)

    ligysis = data.get('LIGYSIS', {})

    # Extract unique PDB_chain combinations
    structures = set()
    for key in ligysis.keys():
        # Key format: "1a52_A_1" -> (1a52, A)
        parts = key.rsplit('_', 2)
        if len(parts) >= 2:
            pdb_id = parts[0].lower()
            chain = parts[1]
            structures.add((pdb_id, chain))

    print(f"Found {len(structures)} unique structures in ground truth")

    if args.quick:
        structures = list(structures)[:100]
        print(f"Quick mode: using first 100 structures")
    else:
        structures = list(structures)

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Download structures
    print(f"\nDownloading structures to {args.output}...")
    print(f"Using {args.parallel} parallel workers")

    success = 0
    failed = 0
    cached = 0

    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = {
            executor.submit(download_structure, pdb, chain, args.output): (pdb, chain)
            for pdb, chain in structures
        }

        for i, future in enumerate(as_completed(futures), 1):
            pdb_id, chain, ok, status = future.result()
            if ok:
                if status == 'cached':
                    cached += 1
                else:
                    success += 1
                if i % 50 == 0:
                    print(f"  Progress: {i}/{len(structures)} ({success} new, {cached} cached, {failed} failed)")
            else:
                failed += 1
                print(f"  Failed: {pdb_id}_{chain}")

    print(f"\nDownload complete:")
    print(f"  New downloads: {success}")
    print(f"  Already cached: {cached}")
    print(f"  Failed: {failed}")
    print(f"  Total available: {success + cached}")
    print(f"\nStructures saved to: {args.output}")


if __name__ == '__main__':
    main()
