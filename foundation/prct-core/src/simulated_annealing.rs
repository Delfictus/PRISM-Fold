//! Simulated Annealing for Graph Coloring Refinement
//!
//! Takes an initial coloring and refines it through simulated annealing
//! to reduce the chromatic number.

use crate::errors::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use shared_types::*;

/// Refine a coloring using simulated annealing
pub fn simulated_annealing_refinement(
    graph: &Graph,
    initial_solution: &ColoringSolution,
    max_iterations: usize,
    initial_temperature: f64,
) -> Result<ColoringSolution> {
    let start = std::time::Instant::now();

    let mut current_coloring = initial_solution.colors.clone();
    let mut current_chromatic = initial_solution.chromatic_number;
    let mut current_conflicts = 0;

    let mut best_coloring = current_coloring.clone();
    let mut best_chromatic = current_chromatic;

    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut temperature = initial_temperature;
    let cooling_rate = 0.9995;

    let n = graph.num_vertices;

    for iter in 0..max_iterations {
        // Try different move types
        let move_type = iter % 3;

        let (new_coloring, new_chromatic, new_conflicts) = match move_type {
            0 => try_recolor_vertex(&current_coloring, graph, &mut rng),
            1 => try_swap_colors(&current_coloring, graph, &mut rng),
            2 => try_kempe_chain_swap(&current_coloring, graph, &mut rng),
            _ => unreachable!(),
        };

        // Compute delta (lower is better)
        let current_cost = current_chromatic * 100 + current_conflicts * 10;
        let new_cost = new_chromatic * 100 + new_conflicts * 10;
        let delta = new_cost as i32 - current_cost as i32;

        // Accept if better, or probabilistically if worse
        if delta < 0 || rng.gen::<f64>() < (-delta as f64 / temperature).exp() {
            current_coloring = new_coloring;
            current_chromatic = new_chromatic;
            current_conflicts = new_conflicts;

            // Update best if valid and better
            if current_conflicts == 0 && current_chromatic < best_chromatic {
                best_coloring = current_coloring.clone();
                best_chromatic = current_chromatic;

                if iter % 1000 == 0 {
                    println!("    🔥 SA iteration {}: {} colors", iter, best_chromatic);
                }
            }
        }

        // Cool down
        temperature *= cooling_rate;

        // Early exit if we've found a very good solution
        if best_chromatic <= initial_solution.chromatic_number * 7 / 10 {
            println!(
                "    ⚡ SA early exit: reached {} colors (30% improvement)",
                best_chromatic
            );
            break;
        }
    }

    let computation_time = start.elapsed().as_secs_f64() * 1000.0;

    // Verify final solution has no conflicts
    let final_conflicts = count_conflicts(&best_coloring, graph);

    Ok(ColoringSolution {
        colors: best_coloring,
        chromatic_number: best_chromatic,
        conflicts: final_conflicts,
        quality_score: 1.0 - (best_chromatic as f64 / initial_solution.chromatic_number as f64),
        computation_time_ms: computation_time,
    })
}

/// Try recoloring a random vertex
fn try_recolor_vertex(
    coloring: &[usize],
    graph: &Graph,
    rng: &mut impl Rng,
) -> (Vec<usize>, usize, usize) {
    let mut new_coloring = coloring.to_vec();
    let n = graph.num_vertices;

    let v = rng.gen_range(0..n);
    let max_color = *coloring.iter().max().unwrap_or(&0);
    let new_color = rng.gen_range(0..=max_color);

    new_coloring[v] = new_color;

    let conflicts = count_conflicts(&new_coloring, graph);
    let chromatic = compute_chromatic_number(&new_coloring);

    (new_coloring, chromatic, conflicts)
}

/// Try swapping two color classes
fn try_swap_colors(
    coloring: &[usize],
    graph: &Graph,
    rng: &mut impl Rng,
) -> (Vec<usize>, usize, usize) {
    let mut new_coloring = coloring.to_vec();
    let max_color = *coloring.iter().max().unwrap_or(&0);

    if max_color == 0 {
        return (new_coloring, 1, 0);
    }

    let c1 = rng.gen_range(0..=max_color);
    let c2 = rng.gen_range(0..=max_color);

    if c1 == c2 {
        let conflicts = count_conflicts(&new_coloring, graph);
        return (new_coloring, max_color + 1, conflicts);
    }

    // Swap all vertices with c1 ↔ c2
    for color in &mut new_coloring {
        if *color == c1 {
            *color = c2;
        } else if *color == c2 {
            *color = c1;
        }
    }

    let conflicts = count_conflicts(&new_coloring, graph);
    let chromatic = compute_chromatic_number(&new_coloring);

    (new_coloring, chromatic, conflicts)
}

/// Try Kempe chain swap (most powerful move)
fn try_kempe_chain_swap(
    coloring: &[usize],
    graph: &Graph,
    rng: &mut impl Rng,
) -> (Vec<usize>, usize, usize) {
    let n = graph.num_vertices;
    let mut new_coloring = coloring.to_vec();

    // Pick random vertex and try to recolor via Kempe chain
    let v = rng.gen_range(0..n);
    let current_color = coloring[v];
    let max_color = *coloring.iter().max().unwrap_or(&0);

    if max_color == 0 {
        return (new_coloring, 1, 0);
    }

    let target_color = rng.gen_range(0..=max_color);

    if current_color == target_color {
        let conflicts = count_conflicts(&new_coloring, graph);
        return (new_coloring, max_color + 1, conflicts);
    }

    // Build Kempe chain for (current_color, target_color)
    let mut chain = vec![false; n];
    let mut stack = vec![v];
    chain[v] = true;

    while let Some(u) = stack.pop() {
        for w in 0..n {
            if graph.adjacency[u * n + w] && !chain[w] {
                let w_color = new_coloring[w];
                if w_color == current_color || w_color == target_color {
                    chain[w] = true;
                    stack.push(w);
                }
            }
        }
    }

    // Swap colors in the chain
    for u in 0..n {
        if chain[u] {
            if new_coloring[u] == current_color {
                new_coloring[u] = target_color;
            } else if new_coloring[u] == target_color {
                new_coloring[u] = current_color;
            }
        }
    }

    let conflicts = count_conflicts(&new_coloring, graph);
    let chromatic = compute_chromatic_number(&new_coloring);

    (new_coloring, chromatic, conflicts)
}

/// Count conflicts in a coloring
fn count_conflicts(coloring: &[usize], graph: &Graph) -> usize {
    let n = graph.num_vertices;
    let mut conflicts = 0;

    for i in 0..n {
        for j in (i + 1)..n {
            if graph.adjacency[i * n + j] && coloring[i] == coloring[j] {
                conflicts += 1;
            }
        }
    }

    conflicts
}

/// Compute chromatic number (max color + 1)
fn compute_chromatic_number(coloring: &[usize]) -> usize {
    coloring.iter().max().map(|&c| c + 1).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_conflicts() {
        // Simple triangle graph
        let graph = Graph {
            num_vertices: 3,
            num_edges: 3,
            edges: vec![(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)],
            adjacency: vec![false, true, true, true, false, true, true, true, false],
            coordinates: None,
        };

        // Valid 3-coloring
        let valid = vec![0, 1, 2];
        assert_eq!(count_conflicts(&valid, &graph), 0);

        // Invalid (all same color)
        let invalid = vec![0, 0, 0];
        assert_eq!(count_conflicts(&invalid, &graph), 3);
    }

    #[test]
    fn test_chromatic_number() {
        assert_eq!(compute_chromatic_number(&vec![0, 1, 2, 1, 0]), 3);
        assert_eq!(compute_chromatic_number(&vec![0, 0, 0]), 1);
        assert_eq!(compute_chromatic_number(&vec![]), 0);
    }
}
