// Simple greedy graph coloring for pocket clustering
extern "C" __global__ void pocket_clustering_kernel(
    const int* row_ptr,
    const int* col_idx,
    int num_vertices,
    int max_colors,
    int* colors
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;

    // Greedy first-fit coloring based on current neighbor assignments
    for (int c = 0; c < max_colors; ++c) {
        bool conflict = false;
        int start = row_ptr[v];
        int end = row_ptr[v + 1];
        for (int k = start; k < end; ++k) {
            int nbr = col_idx[k];
            int nc = colors[nbr];
            if (nc == c) {
                conflict = true;
                break;
            }
        }
        if (!conflict) {
            colors[v] = c;
            return;
        }
    }
    // Fallback if all colors used
    colors[v] = v % max_colors;
}
