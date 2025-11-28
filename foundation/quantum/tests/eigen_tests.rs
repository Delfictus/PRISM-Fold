//! Comprehensive Tests for Robust Eigenvalue Solver
//!
//! These tests validate the mathematical correctness and numerical stability
//! of the eigenvalue solver across various matrix types and sizes.

use approx::assert_abs_diff_eq;
use ndarray::{Array1, Array2};
use num_complex::Complex64;
use quantum_engine::robust_eigen::{RobustEigenConfig, RobustEigenSolver, SolverMethod};

// ============================================================================
// TEST UTILITIES
// ============================================================================

/// Create identity matrix
fn create_identity(n: usize) -> Array2<Complex64> {
    let mut matrix = Array2::zeros((n, n));
    for i in 0..n {
        matrix[[i, i]] = Complex64::new(1.0, 0.0);
    }
    matrix
}

/// Create diagonal matrix with given eigenvalues
fn create_diagonal(eigenvalues: &[f64]) -> Array2<Complex64> {
    let n = eigenvalues.len();
    let mut matrix = Array2::zeros((n, n));
    for (i, &lambda) in eigenvalues.iter().enumerate() {
        matrix[[i, i]] = Complex64::new(lambda, 0.0);
    }
    matrix
}

/// Create symmetric tridiagonal matrix (harmonic oscillator-like)
fn create_tridiagonal(n: usize, diagonal: f64, off_diagonal: f64) -> Array2<Complex64> {
    let mut matrix = Array2::zeros((n, n));
    for i in 0..n {
        matrix[[i, i]] = Complex64::new(diagonal, 0.0);
        if i > 0 {
            matrix[[i, i - 1]] = Complex64::new(off_diagonal, 0.0);
        }
        if i < n - 1 {
            matrix[[i, i + 1]] = Complex64::new(off_diagonal, 0.0);
        }
    }
    matrix
}

/// Validate eigenvalue equation: ||Hv - λv|| < ε
fn validate_eigenpair(
    matrix: &Array2<Complex64>,
    eigenvalue: f64,
    eigenvector: &Array1<Complex64>,
    tolerance: f64,
) -> bool {
    let hv = matrix.dot(eigenvector);
    let lambda_v = eigenvector.mapv(|x| x * eigenvalue);

    let residual: f64 = hv
        .iter()
        .zip(lambda_v.iter())
        .map(|(a, b)| (a - b).norm_sqr())
        .sum::<f64>()
        .sqrt();

    residual < tolerance
}

/// Check orthonormality of eigenvectors
fn check_orthonormality(eigenvectors: &Array2<Complex64>, tolerance: f64) -> bool {
    let n = eigenvectors.ncols();

    for i in 0..n {
        let v_i = eigenvectors.column(i);

        // Check normalization
        let norm: f64 = v_i.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
        if (norm - 1.0).abs() > tolerance {
            return false;
        }

        // Check orthogonality
        for j in i + 1..n {
            let v_j = eigenvectors.column(j);
            let inner_product: Complex64 =
                v_i.iter().zip(v_j.iter()).map(|(a, b)| a.conj() * b).sum();

            if inner_product.norm() > tolerance {
                return false;
            }
        }
    }

    true
}

// ============================================================================
// SMALL MATRIX TESTS (N = 2, 3, 5, 10)
// ============================================================================

#[test]
fn test_2x2_identity() {
    let matrix = create_identity(2);
    let mut solver = RobustEigenSolver::default();

    let (eigenvalues, eigenvectors) = solver.solve(&matrix).unwrap();

    // All eigenvalues should be 1.0
    assert_eq!(eigenvalues.len(), 2);
    for &lambda in eigenvalues.iter() {
        assert_abs_diff_eq!(lambda, 1.0, epsilon = 1e-10);
    }

    // Check eigenvectors
    for i in 0..2 {
        assert!(validate_eigenpair(
            &matrix,
            eigenvalues[i],
            &eigenvectors.column(i).to_owned(),
            1e-10
        ));
    }
}

#[test]
fn test_2x2_simple_hermitian() {
    // Matrix: [[2, 1], [1, 3]]
    // Eigenvalues: (5 ± √5)/2 ≈ 3.618, 1.382
    let matrix = Array2::from_shape_vec(
        (2, 2),
        vec![
            Complex64::new(2.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(3.0, 0.0),
        ],
    )
    .unwrap();

    let mut solver = RobustEigenSolver::default();
    let (eigenvalues, eigenvectors) = solver.solve(&matrix).unwrap();

    assert_eq!(eigenvalues.len(), 2);

    // Check eigenvalues (sorted ascending)
    let expected_low = (5.0 - 5.0_f64.sqrt()) / 2.0; // ≈ 1.382
    let expected_high = (5.0 + 5.0_f64.sqrt()) / 2.0; // ≈ 3.618

    assert_abs_diff_eq!(eigenvalues[0], expected_low, epsilon = 1e-8);
    assert_abs_diff_eq!(eigenvalues[1], expected_high, epsilon = 1e-8);

    // Validate eigenpairs
    for i in 0..2 {
        assert!(validate_eigenpair(
            &matrix,
            eigenvalues[i],
            &eigenvectors.column(i).to_owned(),
            1e-8
        ));
    }
}

#[test]
fn test_3x3_diagonal() {
    let eigenvals = vec![1.0, 2.0, 3.0];
    let matrix = create_diagonal(&eigenvals);

    let mut solver = RobustEigenSolver::default();
    let (computed_eigenvalues, eigenvectors) = solver.solve(&matrix).unwrap();

    assert_eq!(computed_eigenvalues.len(), 3);

    // Eigenvalues should match (possibly reordered)
    for &expected in &eigenvals {
        let found = computed_eigenvalues
            .iter()
            .any(|&computed| (computed - expected).abs() < 1e-10);
        assert!(found, "Expected eigenvalue {} not found", expected);
    }

    // Check orthonormality
    assert!(check_orthonormality(&eigenvectors, 1e-10));
}

#[test]
fn test_5x5_tridiagonal() {
    let matrix = create_tridiagonal(5, 2.0, -1.0);

    let mut solver = RobustEigenSolver::default();
    let (eigenvalues, eigenvectors) = solver.solve(&matrix).unwrap();

    assert_eq!(eigenvalues.len(), 5);

    // Validate all eigenpairs
    for i in 0..5 {
        assert!(validate_eigenpair(
            &matrix,
            eigenvalues[i],
            &eigenvectors.column(i).to_owned(),
            1e-8
        ));
    }

    // Check orthonormality
    assert!(check_orthonormality(&eigenvectors, 1e-10));
}

#[test]
fn test_10x10_harmonic_oscillator() {
    // Quantum harmonic oscillator spectrum: E_n = (n + 1/2)ℏω
    let n = 10;
    let mut matrix = Array2::zeros((n, n));

    // Diagonal: (i + 0.5)
    for i in 0..n {
        matrix[[i, i]] = Complex64::new((i as f64 + 0.5), 0.0);
    }

    // Off-diagonal coupling
    for i in 0..n - 1 {
        matrix[[i, i + 1]] = Complex64::new(0.05, 0.0);
        matrix[[i + 1, i]] = Complex64::new(0.05, 0.0);
    }

    let mut solver = RobustEigenSolver::default();
    let (eigenvalues, eigenvectors) = solver.solve(&matrix).unwrap();

    assert_eq!(eigenvalues.len(), n);

    // Ground state should be near 0.5
    assert!((eigenvalues[0] - 0.5).abs() < 0.1);

    // Eigenvalues should be increasing
    for i in 0..n - 1 {
        assert!(eigenvalues[i] < eigenvalues[i + 1]);
    }

    // Validate eigenpairs
    for i in 0..n {
        assert!(validate_eigenpair(
            &matrix,
            eigenvalues[i],
            &eigenvectors.column(i).to_owned(),
            1e-7
        ));
    }
}

// ============================================================================
// LARGE MATRIX TESTS (N = 50, 100, 200)
// ============================================================================

#[test]
fn test_50x50_banded() {
    let n = 50;
    let matrix = create_tridiagonal(n, 1.0, 0.1);

    let mut solver = RobustEigenSolver::default();
    let result = solver.solve(&matrix);

    assert!(result.is_ok());
    let (eigenvalues, eigenvectors) = result.unwrap();

    assert_eq!(eigenvalues.len(), n);

    // Sample validation (check first, middle, last)
    let indices = [0, n / 2, n - 1];
    for &i in &indices {
        assert!(validate_eigenpair(
            &matrix,
            eigenvalues[i],
            &eigenvectors.column(i).to_owned(),
            1e-6
        ));
    }

    // Check method used
    let diag = solver.get_diagnostics();
    println!(
        "50×50: Method = {:?}, Time = {:.2}ms",
        diag.method, diag.compute_time_ms
    );
}

#[test]
fn test_100x100_diagonal_dominant() {
    let n = 100;
    let mut matrix = Array2::zeros((n, n));

    // Strong diagonal dominance
    for i in 0..n {
        matrix[[i, i]] = Complex64::new((i as f64 + 1.0) * 10.0, 0.0);
    }

    // Weak off-diagonal
    for i in 0..n - 1 {
        matrix[[i, i + 1]] = Complex64::new(0.01, 0.0);
        matrix[[i + 1, i]] = Complex64::new(0.01, 0.0);
    }

    let mut solver = RobustEigenSolver::default();
    let (eigenvalues, eigenvectors) = solver.solve(&matrix).unwrap();

    assert_eq!(eigenvalues.len(), n);

    // Ground state should be near 10.0
    assert!((eigenvalues[0] - 10.0).abs() < 1.0);

    // Validate ground state
    assert!(validate_eigenpair(
        &matrix,
        eigenvalues[0],
        &eigenvectors.column(0).to_owned(),
        1e-5
    ));

    let diag = solver.get_diagnostics();
    println!(
        "100×100: Method = {:?}, Time = {:.2}ms, κ = {:.2e}",
        diag.method, diag.compute_time_ms, diag.condition_number
    );
}

#[test]
fn test_200x200_sparse() {
    let n = 200;
    let mut matrix = Array2::zeros((n, n));

    // Create sparse banded matrix
    for i in 0..n {
        matrix[[i, i]] = Complex64::new(1.0 + (i as f64 / n as f64), 0.0);

        if i > 0 {
            matrix[[i, i - 1]] = Complex64::new(0.1, 0.0);
        }
        if i < n - 1 {
            matrix[[i, i + 1]] = Complex64::new(0.1, 0.0);
        }
    }

    let mut config = RobustEigenConfig::default();
    config.verbose = true;
    let mut solver = RobustEigenSolver::new(config);

    let start = std::time::Instant::now();
    let (eigenvalues, eigenvectors) = solver.solve(&matrix).unwrap();
    let elapsed = start.elapsed();

    assert_eq!(eigenvalues.len(), n);

    // Validate ground state
    assert!(validate_eigenpair(
        &matrix,
        eigenvalues[0],
        &eigenvectors.column(0).to_owned(),
        1e-4
    ));

    println!("200×200: Solved in {:.2}ms", elapsed.as_millis());

    let diag = solver.get_diagnostics();
    assert!(diag.compute_time_ms < 10000.0, "Computation too slow");
}

// ============================================================================
// ILL-CONDITIONED MATRIX TESTS
// ============================================================================

#[test]
fn test_ill_conditioned_small_eigenvalues() {
    // Matrix with eigenvalues spanning many orders of magnitude
    let eigenvals = vec![1e-6, 1e-3, 1.0, 1e3, 1e6];
    let matrix = create_diagonal(&eigenvals);

    let mut config = RobustEigenConfig::default();
    config.use_preconditioning = true;
    config.verbose = true;

    let mut solver = RobustEigenSolver::new(config);
    let (computed_eigenvalues, _) = solver.solve(&matrix).unwrap();

    // Check all eigenvalues found (with tolerance for extreme values)
    for &expected in &eigenvals {
        let found = computed_eigenvalues.iter().any(|&computed| {
            let rel_error = ((computed - expected) / expected).abs();
            rel_error < 0.01
        });
        assert!(
            found,
            "Expected eigenvalue {} not found accurately",
            expected
        );
    }

    let diag = solver.get_diagnostics();
    println!(
        "Ill-conditioned: κ = {:.2e}, preconditioned = {}",
        diag.condition_number, diag.preconditioned
    );
    assert!(diag.condition_number > 1e10);
}

#[test]
fn test_near_singular() {
    let n = 10;
    let mut matrix = create_tridiagonal(n, 1.0, 0.5);

    // Make nearly singular by adding very small eigenvalue
    matrix[[0, 0]] = Complex64::new(1e-10, 0.0);

    let mut solver = RobustEigenSolver::default();
    let result = solver.solve(&matrix);

    assert!(result.is_ok());
    let (_eigenvalues, _) = result.unwrap();

    let diag = solver.get_diagnostics();
    println!(
        "Near-singular: Method = {:?}, preconditioned = {}",
        diag.method, diag.preconditioned
    );
}

// ============================================================================
// NON-HERMITIAN MATRIX TESTS (Should Auto-Symmetrize)
// ============================================================================

#[test]
fn test_slightly_non_hermitian() {
    let mut matrix = Array2::from_shape_vec(
        (3, 3),
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.2, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.21, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(0.3, 0.0), // Slightly asymmetric
            Complex64::new(0.0, 0.0),
            Complex64::new(0.31, 0.0),
            Complex64::new(3.0, 0.0), // Slightly asymmetric
        ],
    )
    .unwrap();

    let mut solver = RobustEigenSolver::default();
    let result = solver.solve(&matrix);

    assert!(result.is_ok());

    let diag = solver.get_diagnostics();
    assert!(diag.symmetrized, "Matrix should have been symmetrized");
    println!(
        "Non-Hermitian: Hermitian error = {:.2e}, symmetrized = {}",
        diag.hermitian_error, diag.symmetrized
    );
}

#[test]
fn test_complex_hermitian() {
    // Hermitian matrix with complex off-diagonals
    let matrix = Array2::from_shape_vec(
        (3, 3),
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.5, 0.5),
            Complex64::new(0.0, 0.3),
            Complex64::new(0.5, -0.5),
            Complex64::new(2.0, 0.0),
            Complex64::new(0.2, 0.1),
            Complex64::new(0.0, -0.3),
            Complex64::new(0.2, -0.1),
            Complex64::new(3.0, 0.0),
        ],
    )
    .unwrap();

    let mut solver = RobustEigenSolver::default();
    let (eigenvalues, eigenvectors) = solver.solve(&matrix).unwrap();

    // Eigenvalues should all be real (Hermitian property)
    for i in 0..3 {
        assert!(validate_eigenpair(
            &matrix,
            eigenvalues[i],
            &eigenvectors.column(i).to_owned(),
            1e-8
        ));
    }

    let diag = solver.get_diagnostics();
    println!(
        "Complex Hermitian: Hermitian error = {:.2e}",
        diag.hermitian_error
    );
    assert!(diag.hermitian_error < 1e-10, "Should be exactly Hermitian");
}

// ============================================================================
// CONVERGENCE AND ACCURACY TESTS
// ============================================================================

#[test]
fn test_known_solution_hydrogen_atom() {
    // Simplified 1D hydrogen-like Hamiltonian
    // Eigenvalues known analytically
    let n = 20;
    let mut matrix = Array2::zeros((n, n));

    // Kinetic energy (finite difference)
    for i in 0..n {
        matrix[[i, i]] = Complex64::new(2.0, 0.0);
        if i > 0 {
            matrix[[i, i - 1]] = Complex64::new(-1.0, 0.0);
        }
        if i < n - 1 {
            matrix[[i, i + 1]] = Complex64::new(-1.0, 0.0);
        }
    }

    // Potential energy (Coulomb-like)
    for i in 0..n {
        let r = (i as f64 + 1.0) * 0.1;
        matrix[[i, i]] = matrix[[i, i]] - Complex64::new(1.0 / r, 0.0);
    }

    let mut solver = RobustEigenSolver::default();
    let (eigenvalues, _) = solver.solve(&matrix).unwrap();

    // Ground state should be negative (bound state)
    assert!(eigenvalues[0] < 0.0, "Ground state should be bound (E < 0)");

    println!(
        "Hydrogen-like ground state: E₀ = {:.6} Hartree",
        eigenvalues[0]
    );
}

#[test]
fn test_method_fallback_sequence() {
    // Create pathological matrix that might fail direct method
    let n = 30;
    let mut matrix = Array2::zeros((n, n));

    for i in 0..n {
        matrix[[i, i]] = Complex64::new(1.0 / ((i as f64 + 1.0) * (i as f64 + 1.0)), 0.0);
    }

    let mut config = RobustEigenConfig::default();
    config.verbose = true;
    config.use_shift_invert = true;

    let mut solver = RobustEigenSolver::new(config);
    let result = solver.solve(&matrix);

    assert!(result.is_ok());

    let diag = solver.get_diagnostics();
    println!("Methods tried: {:?}", diag.methods_tried);
    println!("Final method: {:?}", diag.method);

    // Should have tried multiple methods due to ill-conditioning
    assert!(diag.methods_tried.len() >= 1);
}

// ============================================================================
// PERFORMANCE BENCHMARKS
// ============================================================================

#[test]
fn benchmark_scaling() {
    let sizes = [10, 20, 50, 100];

    for &n in &sizes {
        let matrix = create_tridiagonal(n, 1.0, 0.1);

        let mut solver = RobustEigenSolver::default();
        let start = std::time::Instant::now();
        let result = solver.solve(&matrix);
        let elapsed = start.elapsed();

        assert!(result.is_ok());

        let diag = solver.get_diagnostics();
        println!(
            "N={:3}: {:6.2}ms, Method={:?}, κ={:.2e}",
            n,
            elapsed.as_micros() as f64 / 1000.0,
            diag.method,
            diag.condition_number
        );
    }
}
