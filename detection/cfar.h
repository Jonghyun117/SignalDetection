/* detection/cfar.h — CA-CFAR (Cell Averaging CFAR) interface */
#ifndef CFAR_H
#define CFAR_H

#include <stdint.h>

/* Parameters for one CFAR pass. */
typedef struct {
    int   n_guard;   /* guard cells each side of CUT (skip these)          */
    int   n_ref;     /* reference cells each side                           */
    float alpha;     /* threshold = alpha * noise_estimate                  */
} CfarParams;

/* Compute alpha for a target false-alarm probability over N_ref cells.
 *   alpha = N_ref * (pfa^(-1/N_ref) - 1)           (CA-CFAR theory)
 * n_ref: reference cells per side (total used = 2*n_ref). */
float cfar_alpha(int n_ref, float pfa);

/* Run 1-D CA-CFAR on a linear power array.
 *
 * power[n]   : linear power values (float, length n_pts)
 * n_pts      : length of power array
 * params     : CFAR configuration
 * detections : output array of detected bin indices (caller-allocated)
 * max_det    : capacity of detections[]
 *
 * Returns number of detected bins (≤ max_det). */
int cfar_1d(const float *power, int n_pts, const CfarParams *params,
            int *detections, int max_det);

/* Cluster adjacent CFAR detections separated by ≤ gap_bins into groups.
 *
 * dets[]     : sorted array of detected bin indices (length n_dets)
 * gap_bins   : maximum gap to merge (inclusive)
 * starts[]   : output: first bin of each cluster
 * ends[]     : output: last  bin of each cluster
 * max_clust  : capacity of starts[]/ends[]
 *
 * Returns number of clusters. */
int cfar_cluster(const int *dets, int n_dets, int gap_bins,
                 int *starts, int *ends, int max_clust);

#endif /* CFAR_H */
