//! CPU Thread Pool Initialization
//!
//! Central initialization for Rayon thread pool with controlled concurrency.
//! Ensures single initialization per process, with environment variable override.

use std::sync::Once;

static INIT: Once = Once::new();

/// Initialize Rayon global thread pool with specified thread count.
///
/// # Behavior
/// - Checks `RAYON_NUM_THREADS` environment variable first (highest priority)
/// - If env var is set, uses that value and ignores `num_threads` parameter
/// - Otherwise, initializes pool with `num_threads` parameter
/// - Uses `std::sync::Once` to ensure single initialization per process
/// - Safe to call multiple times (subsequent calls are no-ops)
///
/// # Arguments
/// * `num_threads` - Desired thread count (ignored if RAYON_NUM_THREADS is set)
///
/// # Environment Variables
/// * `RAYON_NUM_THREADS` - Override thread count (e.g., `RAYON_NUM_THREADS=24`)
///
/// # Example
/// ```no_run
/// use prct_core::cpu_init::init_rayon_threads;
///
/// // Initialize with 24 threads (unless RAYON_NUM_THREADS is set)
/// init_rayon_threads(24);
/// ```
pub fn init_rayon_threads(num_threads: usize) {
    INIT.call_once(|| {
        // Check if environment variable overrides config
        if let Ok(env_threads) = std::env::var("RAYON_NUM_THREADS") {
            println!(
                "[CPU] Rayon threads controlled by RAYON_NUM_THREADS env: {}",
                env_threads
            );
            return; // Rayon will read RAYON_NUM_THREADS directly
        }

        // Initialize with config value
        match rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_global()
        {
            Ok(_) => {
                println!(
                    "[CPU] Initialized Rayon thread pool: {} threads",
                    num_threads
                );
            }
            Err(e) => {
                eprintln!("[CPU] Warning: Failed to initialize Rayon pool: {}", e);
                eprintln!("[CPU] Using default Rayon configuration");
            }
        }
    });
}
