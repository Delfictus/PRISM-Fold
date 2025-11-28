//! Graph Problem Domain Types
//!
//! Data structures for graph coloring and TSP problems

/// Graph representation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Graph {
    /// Number of vertices
    pub num_vertices: usize,

    /// Number of edges
    pub num_edges: usize,

    /// Edge list (source, target, weight)
    pub edges: alloc::vec::Vec<(usize, usize, f64)>,

    /// Adjacency matrix (flattened, row-major)
    pub adjacency: alloc::vec::Vec<bool>,

    /// Vertex coordinates (for TSP, optional)
    pub coordinates: Option<alloc::vec::Vec<(f64, f64)>>,
}

/// Graph coloring solution
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ColoringSolution {
    /// Color assignment for each vertex
    pub colors: alloc::vec::Vec<usize>,

    /// Number of colors used (chromatic number)
    pub chromatic_number: usize,

    /// Number of conflicts (0 = valid coloring)
    pub conflicts: usize,

    /// Solution quality score (higher = better)
    pub quality_score: f64,

    /// Computation time (milliseconds)
    pub computation_time_ms: f64,
}

/// TSP tour solution
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TSPSolution {
    /// Vertex visit order (permutation)
    pub tour: alloc::vec::Vec<usize>,

    /// Total tour length/cost
    pub tour_length: f64,

    /// Number of vertices in tour
    pub num_vertices: usize,

    /// Solution quality score (lower length = better)
    pub quality_score: f64,

    /// Computation time (milliseconds)
    pub computation_time_ms: f64,
}

/// Combined PRCT solution (coloring + TSP)
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PRCTSolution {
    /// Graph coloring solution
    pub coloring: ColoringSolution,

    /// TSP tour within each color class
    pub color_class_tours: alloc::vec::Vec<TSPSolution>,

    /// Phase coherence used in solution
    pub phase_coherence: f64,

    /// Kuramoto order parameter achieved
    pub kuramoto_order: f64,

    /// Overall solution quality
    pub overall_quality: f64,

    /// Total computation time (milliseconds)
    pub total_time_ms: f64,
}

extern crate alloc;
