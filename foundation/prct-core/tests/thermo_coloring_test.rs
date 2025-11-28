//! Unit test for thermodynamic coloring chromatic preservation
//!
//! Tests that phase-to-color conversion doesn't collapse chromatic number

#[cfg(test)]
mod tests {
    use shared_types::*;

    #[test]
    fn test_phase_to_color_preserves_chromatic() {
        // Create a 127-color solution (simulating Phase 1 output)
        let n = 1000;
        let initial_chromatic = 127;
        let colors: Vec<usize> = (0..n).map(|v| v % initial_chromatic).collect();

        let initial = ColoringSolution {
            colors: colors.clone(),
            chromatic_number: initial_chromatic,
            conflicts: 0,
            quality_score: 1.0,
            computation_time_ms: 0.0,
        };

        // Simulate the OLD BUGGY phase-to-color conversion (using target_chromatic)
        let target_chromatic = 83; // World record goal
        let buggy_colors: Vec<usize> = initial
            .colors
            .iter()
            .map(|&c| {
                let phase = (c as f32 / target_chromatic as f32) * 2.0 * std::f32::consts::PI;
                let normalized =
                    (phase.rem_euclid(2.0 * std::f32::consts::PI)) / (2.0 * std::f32::consts::PI);
                (normalized * target_chromatic as f32).floor() as usize % target_chromatic
            })
            .collect();

        let buggy_chromatic = buggy_colors.iter().max().unwrap_or(&0) + 1;

        // This should demonstrate the bug: chromatic collapses to ~19
        assert!(
            buggy_chromatic < 30,
            "Bug not reproduced! Expected chromatic ~19, got {}",
            buggy_chromatic
        );

        // Simulate the FIXED phase-to-color conversion (using initial_chromatic + slack)
        let color_range = initial_chromatic + 20; // 147
        let fixed_colors: Vec<usize> = initial
            .colors
            .iter()
            .map(|&c| {
                let phase = (c as f32 / color_range as f32) * 2.0 * std::f32::consts::PI;
                let normalized =
                    (phase.rem_euclid(2.0 * std::f32::consts::PI)) / (2.0 * std::f32::consts::PI);
                (normalized * color_range as f32).floor() as usize % color_range
            })
            .collect();

        // Compact colors (renumber to sequential)
        use std::collections::HashMap;
        let mut color_map: HashMap<usize, usize> = HashMap::new();
        let mut next_color = 0;
        let mut compacted_colors = fixed_colors.clone();

        for c in &mut compacted_colors {
            let new_color = *color_map.entry(*c).or_insert_with(|| {
                let nc = next_color;
                next_color += 1;
                nc
            });
            *c = new_color;
        }

        let fixed_chromatic = next_color;

        // Fixed version should preserve chromatic around 100-130
        assert!(
            fixed_chromatic >= 100,
            "Chromatic collapsed to {} (expected ≥100)",
            fixed_chromatic
        );
        assert!(
            fixed_chromatic <= 130,
            "Chromatic exploded to {} (expected ≤130)",
            fixed_chromatic
        );

        println!("[TEST] Buggy chromatic: {}", buggy_chromatic);
        println!("[TEST] Fixed chromatic: {}", fixed_chromatic);
        println!("[TEST] Color range: {}", color_range);
        println!(
            "[TEST] Compaction ratio: {:.3}",
            fixed_chromatic as f64 / color_range as f64
        );
    }

    #[test]
    fn test_color_compaction() {
        // Test that compaction removes gaps correctly
        let sparse_colors = vec![0, 5, 5, 10, 10, 20, 20, 25];

        use std::collections::HashMap;
        let mut color_map: HashMap<usize, usize> = HashMap::new();
        let mut next_color = 0;
        let mut compacted: Vec<usize> = sparse_colors.clone();

        for c in &mut compacted {
            let new_color = *color_map.entry(*c).or_insert_with(|| {
                let nc = next_color;
                next_color += 1;
                nc
            });
            *c = new_color;
        }

        // Should have 4 unique colors (0, 5, 10, 20, 25 -> 5 unique -> wait, let me recount)
        // Unique: {0, 5, 10, 20, 25} = 5 colors
        assert_eq!(
            next_color, 5,
            "Should have 5 unique colors after compaction"
        );

        // Verify sequential [0..5)
        let mut unique: Vec<usize> = compacted.iter().copied().collect();
        unique.sort();
        unique.dedup();
        assert_eq!(unique, vec![0, 1, 2, 3, 4], "Colors should be sequential");
    }
}
