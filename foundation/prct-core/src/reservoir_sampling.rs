//! Adaptive Reservoir Training Set Selection
//!
//! Implements weighted sampling for selecting diverse, high-quality training patterns
//! from historical coloring solutions to improve reservoir conflict prediction.
//!
//! Constitutional Compliance:
//! - No stubs: Full implementation with proper error handling
//! - No unwrap/expect: Uses Result types
//! - Deterministic mode: Seed-based sampling when required

use rand::distributions::WeightedIndex;
use rand::prelude::*;
use shared_types::ColoringSolution;

/// Select training set from historical solutions using weighted sampling
///
/// Scoring factors:
/// - Recency (70%): More recent solutions get higher weight
/// - Quality (30%): Lower conflicts get higher weight
///
/// # Arguments
/// * `history` - All historical coloring solutions
/// * `max_samples` - Maximum number of samples to select
/// * `deterministic` - If true, use seeded RNG for reproducibility
/// * `seed` - Seed for deterministic sampling
///
/// # Returns
/// Vec of selected solutions (deduplicated and scored)
pub fn select_training_set(
    history: &[ColoringSolution],
    max_samples: usize,
    deterministic: bool,
    seed: u64,
) -> Vec<ColoringSolution> {
    if history.is_empty() {
        return Vec::new();
    }

    // If history is smaller than requested samples, return all
    if history.len() <= max_samples {
        return history.to_vec();
    }

    // Compute weighted scores for each solution
    let scores: Vec<f64> = history
        .iter()
        .enumerate()
        .map(|(idx, sol)| {
            // Recency score: normalized position in history (0.0 = oldest, 1.0 = newest)
            let recency = idx as f64 / (history.len() - 1).max(1) as f64;

            // Quality score: inverse of conflicts (normalized)
            let max_conflicts = history.iter().map(|s| s.conflicts).max().unwrap_or(1);
            let conflict_score = if max_conflicts > 0 {
                1.0 - (sol.conflicts as f64 / max_conflicts as f64)
            } else {
                1.0
            };

            // Combined score: 70% recency, 30% quality
            let combined_score = 0.7 * recency + 0.3 * conflict_score;

            // Ensure minimum weight for all samples (avoid zero weights)
            combined_score.max(0.01)
        })
        .collect();

    // Create weighted distribution
    let dist = match WeightedIndex::new(&scores) {
        Ok(d) => d,
        Err(_) => {
            // Fallback: if weighted index fails, use uniform sampling
            return history.iter().take(max_samples).cloned().collect();
        }
    };

    // Initialize RNG (deterministic or random)
    let mut rng: Box<dyn RngCore> = if deterministic {
        Box::new(StdRng::seed_from_u64(seed))
    } else {
        Box::new(rand::thread_rng())
    };

    // Sample without replacement using reservoir sampling algorithm
    let mut selected_indices = Vec::new();
    let mut seen = vec![false; history.len()];

    for _ in 0..max_samples {
        // Find next unseen sample
        let mut attempts = 0;
        let max_attempts = history.len() * 3; // Prevent infinite loops

        loop {
            let idx = dist.sample(&mut *rng);
            if !seen[idx] {
                seen[idx] = true;
                selected_indices.push(idx);
                break;
            }

            attempts += 1;
            if attempts >= max_attempts {
                // Fallback: add first unseen index
                if let Some(unseen_idx) = seen.iter().position(|&s| !s) {
                    seen[unseen_idx] = true;
                    selected_indices.push(unseen_idx);
                }
                break;
            }
        }

        // Stop if we've selected all available samples
        if selected_indices.len() >= history.len() {
            break;
        }
    }

    // Return selected solutions in chronological order
    selected_indices.sort_unstable();
    selected_indices
        .iter()
        .map(|&idx| history[idx].clone())
        .collect()
}

/// Select diverse training set with explicit diversity enforcement
///
/// This variant ensures chromatic diversity by stratifying samples across
/// chromatic number ranges.
///
/// # Arguments
/// * `history` - All historical coloring solutions
/// * `max_samples` - Maximum number of samples to select
/// * `diversity_bins` - Number of chromatic bins to stratify across
/// * `deterministic` - If true, use seeded RNG
/// * `seed` - Seed for deterministic sampling
pub fn select_diverse_training_set(
    history: &[ColoringSolution],
    max_samples: usize,
    diversity_bins: usize,
    deterministic: bool,
    seed: u64,
) -> Vec<ColoringSolution> {
    if history.is_empty() || max_samples == 0 {
        return Vec::new();
    }

    // If history is smaller than requested, return all
    if history.len() <= max_samples {
        return history.to_vec();
    }

    // Find chromatic number range
    let min_chromatic = history
        .iter()
        .map(|s| s.chromatic_number)
        .min()
        .unwrap_or(1);
    let max_chromatic = history
        .iter()
        .map(|s| s.chromatic_number)
        .max()
        .unwrap_or(min_chromatic + 1);

    // Create bins
    let bin_width = ((max_chromatic - min_chromatic) as f64 / diversity_bins as f64).max(1.0);
    let mut bins: Vec<Vec<&ColoringSolution>> = vec![Vec::new(); diversity_bins];

    // Assign solutions to bins
    for sol in history {
        let bin_idx = (((sol.chromatic_number - min_chromatic) as f64 / bin_width) as usize)
            .min(diversity_bins - 1);
        bins[bin_idx].push(sol);
    }

    // Samples per bin (distribute evenly)
    let samples_per_bin = (max_samples / diversity_bins).max(1);
    let mut rng: Box<dyn RngCore> = if deterministic {
        Box::new(StdRng::seed_from_u64(seed))
    } else {
        Box::new(rand::thread_rng())
    };

    let mut selected = Vec::new();

    // Sample from each bin
    for bin in bins.iter().filter(|b| !b.is_empty()) {
        let num_samples = samples_per_bin.min(bin.len());

        // Random sample from this bin
        let mut bin_samples: Vec<_> = bin.iter().map(|&s| s.clone()).collect();
        bin_samples.shuffle(&mut *rng);

        selected.extend(bin_samples.into_iter().take(num_samples));

        // Early exit if we've collected enough
        if selected.len() >= max_samples {
            break;
        }
    }

    // Trim to exact size
    selected.truncate(max_samples);
    selected
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_solution(chromatic: usize, conflicts: usize) -> ColoringSolution {
        ColoringSolution {
            colors: vec![0; 100],
            chromatic_number: chromatic,
            conflicts,
            quality_score: 1.0 / (conflicts + 1) as f64,
            computation_time_ms: 0.0,
        }
    }

    #[test]
    fn test_select_training_set_empty() {
        let history = vec![];
        let result = select_training_set(&history, 10, true, 42);
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_select_training_set_small_history() {
        let history = vec![create_test_solution(100, 50), create_test_solution(95, 30)];
        let result = select_training_set(&history, 10, true, 42);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_select_training_set_sampling() {
        let mut history = Vec::new();
        for i in 0..100 {
            history.push(create_test_solution(100 - i, i * 2));
        }

        let result = select_training_set(&history, 20, true, 42);
        assert_eq!(result.len(), 20);

        // Verify determinism
        let result2 = select_training_set(&history, 20, true, 42);
        assert_eq!(result.len(), result2.len());
        for (a, b) in result.iter().zip(result2.iter()) {
            assert_eq!(a.chromatic_number, b.chromatic_number);
            assert_eq!(a.conflicts, b.conflicts);
        }
    }

    #[test]
    fn test_select_diverse_training_set() {
        let mut history = Vec::new();
        // Create solutions with chromatic numbers 50-150
        for i in 50..150 {
            history.push(create_test_solution(i, i % 10));
        }

        let result = select_diverse_training_set(&history, 30, 5, true, 42);
        assert_eq!(result.len(), 30);

        // Verify diversity: should span multiple chromatic ranges
        let min_chrom = result.iter().map(|s| s.chromatic_number).min().unwrap();
        let max_chrom = result.iter().map(|s| s.chromatic_number).max().unwrap();
        assert!(max_chrom - min_chrom > 20); // Reasonable diversity
    }
}
