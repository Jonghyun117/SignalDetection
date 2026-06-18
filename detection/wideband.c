/* detection/wideband.c — Wideband spectrum processing + global-threshold detection
 *
 * Input format: WbDmaBin[4096]
 *   i[ch] = uint16 linear power value  (noise 3-10, signal up to ~30000)
 *   q[ch] = uint16, always 0
 *
 * Noise floor: 30th percentile of strided power samples (robust to wide signals).
 * For Exp-distributed power: lambda = perc30 / (-ln 0.70)
 *                            threshold = lambda * (-ln Pfa)
 */
#include "wideband.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define NOISE_SAMPLE_N   1024
#define NOISE_PERCENTILE 0.30f

static int float_cmp(const void *a, const void *b)
{
    float fa = *(const float *)a, fb = *(const float *)b;
    return (fa > fb) - (fa < fb);
}

/* Unpack WbDmaBin[4096] → power[32768].
 * I channel contains the linear power value directly (Q=0, unused). */
static void unpack_power(const WbDmaBin *dma, float *power)
{
    for (int lb = 0; lb < WB_BINS_PER_CH; lb++) {
        for (int ch = 0; ch < WB_N_CHANNELS; ch++) {
            power[ch * WB_BINS_PER_CH + lb] = (float)dma[lb].i[ch];
        }
    }
}

static float _wb_sample[NOISE_SAMPLE_N];

static float estimate_noise_floor(const float *power)
{
    int stride = WB_TOTAL_BINS / NOISE_SAMPLE_N;
    for (int i = 0; i < NOISE_SAMPLE_N; i++)
        _wb_sample[i] = power[i * stride];
    qsort(_wb_sample, NOISE_SAMPLE_N, sizeof(float), float_cmp);
    return _wb_sample[(int)(NOISE_PERCENTILE * NOISE_SAMPLE_N)];
}

static float noise_to_threshold(float perc30, float pfa)
{
    float lambda = perc30 / (-logf(0.70f));
    return lambda * (-logf(pfa));
}

static float centroid_bin(const float *power, int start, int end)
{
    float sw = 0.0f, swk = 0.0f;
    for (int k = start; k <= end; k++) { sw += power[k]; swk += power[k] * k; }
    return (sw > 0.0f) ? swk / sw : (float)((start + end) / 2);
}

static float peak_power(const float *power, int start, int end, int *pk_bin)
{
    float best = power[start]; *pk_bin = start;
    for (int k = start + 1; k <= end; k++)
        if (power[k] > best) { best = power[k]; *pk_bin = k; }
    return best;
}

/* Power in "raw units²" → dBFS relative to uint16 full-scale (65535²) */
static float power_to_dbfs(float raw_power)
{
    return 10.0f * log10f(raw_power / (65535.0f * 65535.0f) + 1e-30f);
}

/* ── Static work buffers ─────────────────────────────────────────────── */
static float _wb_power[WB_TOTAL_BINS];
static int   _wb_dets[WB_TOTAL_BINS];
static int   _wb_starts[DETECT_MAX_SIGNALS];
static int   _wb_ends[DETECT_MAX_SIGNALS];

/* ── Public API ──────────────────────────────────────────────────────── */

float wb_bin_to_freq(int global_bin, float rf_center_hz)
{
    float offset = ((float)global_bin - (float)(WB_TOTAL_BINS / 2)) * WB_FREQ_RES_HZ;
    return rf_center_hz + offset;
}

int wb_freq_to_bin(float freq_hz, float rf_center_hz)
{
    float offset = freq_hz - rf_center_hz;
    int   bin    = (int)(offset / WB_FREQ_RES_HZ + (float)(WB_TOTAL_BINS / 2) + 0.5f);
    if (bin < 0 || bin >= WB_TOTAL_BINS) return -1;
    return bin;
}

void wb_detect(const WbDmaBin *dma_buf, float rf_center_hz,
               const CfarParams *params, SignalList *out)
{
    float pfa      = params ? params->alpha : WB_CFAR_PFA;
    int   min_bins = WB_MIN_CLUSTER_BINS;

    unpack_power(dma_buf, _wb_power);

    float perc30    = estimate_noise_floor(_wb_power);
    float threshold = noise_to_threshold(perc30, pfa);

    int n_dets = 0;
    for (int k = 0; k < WB_TOTAL_BINS && n_dets < WB_TOTAL_BINS; k++)
        if (_wb_power[k] > threshold)
            _wb_dets[n_dets++] = k;

    int gap     = WB_CFAR_N_GUARD * 2;
    int n_clust = cfar_cluster(_wb_dets, n_dets, gap,
                               _wb_starts, _wb_ends, DETECT_MAX_SIGNALS);

    out->n_signals = 0;
    for (int c = 0; c < n_clust && out->n_signals < DETECT_MAX_SIGNALS; c++) {
        int start = _wb_starts[c], end = _wb_ends[c];
        if (end - start + 1 < min_bins) continue;

        int   pk_bin;
        float pk_pwr = peak_power(_wb_power, start, end, &pk_bin);
        float cf_bin = centroid_bin(_wb_power, start, end);
        float cf_hz  = wb_bin_to_freq((int)(cf_bin + 0.5f), rf_center_hz);
        float bw_hz  = (float)(end - start + 1) * WB_FREQ_RES_HZ;

        Signal *s = &out->signals[out->n_signals++];
        s->center_freq_hz = cf_hz;
        s->bandwidth_hz   = bw_hz;
        s->power_dbfs     = power_to_dbfs(pk_pwr);
        s->wb_peak_bin    = pk_bin;
    }
}
