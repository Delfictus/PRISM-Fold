// GPU Shrake-Rupley style surface accessibility
extern "C" __global__ void surface_accessibility_kernel(
    const float* x,
    const float* y,
    const float* z,
    const float* radii,
    int num_atoms,
    int samples,
    float probe_radius,
    float* out_sasa,
    unsigned char* out_surface
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_atoms) return;

    float radius = radii[idx] + probe_radius;
    float radius2 = radius * radius;
    int exposed = 0;

    const float golden = 3.14159265f * (3.0f - sqrtf(5.0f));
    for (int s = 0; s < samples; ++s) {
        float yk = 1.0f - (2.0f * s) / max(1, samples - 1);
        float r = sqrtf(max(0.0f, 1.0f - yk * yk));
        float theta = golden * s;
        float sx = cosf(theta) * r;
        float sz = sinf(theta) * r;

        float px = x[idx] + sx * radius;
        float py = y[idx] + yk * radius;
        float pz = z[idx] + sz * radius;

        bool occluded = false;
        for (int j = 0; j < num_atoms; ++j) {
            if (j == idx) continue;
            float dx = px - x[j];
            float dy = py - y[j];
            float dz = pz - z[j];
            float rj = radii[j] + probe_radius;
            float cutoff2 = rj * rj;
            float d2 = dx * dx + dy * dy + dz * dz;
            if (d2 <= cutoff2) {
                occluded = true;
                break;
            }
        }
        if (!occluded) {
            exposed++;
        }
    }

    float frac = exposed / (float)max(1, samples);
    out_sasa[idx] = 4.0f * 3.14159265f * radius2 * frac;
    if (out_surface) {
        out_surface[idx] = (unsigned char)(frac > 0.05f);
    }
}
