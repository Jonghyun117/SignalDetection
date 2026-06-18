/* detection/cfar.c — Cell Averaging CFAR implementation */
#include "cfar.h"
#include <math.h>
#include <string.h>

float cfar_alpha(int n_ref, float pfa)
{
    /* alpha = 2*n_ref * (pfa^(-1/(2*n_ref)) - 1)
     * Using 2*n_ref total reference cells (n_ref each side). */
    int total = 2 * n_ref;
    return (float)total * (powf(pfa, -1.0f / (float)total) - 1.0f);
}

int cfar_1d(const float *power, int n_pts, const CfarParams *params,
            int *detections, int max_det)
{
    int   G   = params->n_guard;
    int   R   = params->n_ref;
    float alpha = params->alpha;
    int   n_det = 0;

    /* Sliding window: CUT at index i, reference window [i-G-R .. i-G-1]
     * and [i+G+1 .. i+G+R], skipping guard cells [i-G .. i+G]. */
    for (int i = 0; i < n_pts; i++) {
        int left_start  = i - G - R;
        int left_end    = i - G - 1;
        int right_start = i + G + 1;
        int right_end   = i + G + R;

        /* Clip to valid range — use only available cells. */
        int n_cells = 0;
        float noise_sum = 0.0f;

        for (int k = left_start; k <= left_end; k++) {
            if (k >= 0 && k < n_pts) { noise_sum += power[k]; n_cells++; }
        }
        for (int k = right_start; k <= right_end; k++) {
            if (k >= 0 && k < n_pts) { noise_sum += power[k]; n_cells++; }
        }

        if (n_cells == 0) continue;

        float noise_est = noise_sum / (float)n_cells;
        float threshold = alpha * noise_est;

        if (power[i] > threshold) {
            if (n_det < max_det)
                detections[n_det++] = i;
        }
    }
    return n_det;
}

int cfar_cluster(const int *dets, int n_dets, int gap_bins,
                 int *starts, int *ends, int max_clust)
{
    if (n_dets == 0) return 0;

    int n_clust = 0;
    starts[0] = dets[0];
    ends[0]   = dets[0];

    for (int i = 1; i < n_dets; i++) {
        if (dets[i] - ends[n_clust] <= gap_bins + 1) {
            ends[n_clust] = dets[i];
        } else {
            n_clust++;
            if (n_clust >= max_clust) break;
            starts[n_clust] = dets[i];
            ends[n_clust]   = dets[i];
        }
    }
    return n_clust + 1;
}
