/* detection/wideband.c — Wideband spectrum processing + global-threshold detection
 *
 * CA-CFAR sliding window is NOT used for wideband because signals can span
 * hundreds of bins (1 MHz / 18.75 kHz ≈ 53 bins), causing severe masking when
 * reference cells fall inside the signal.
 *
 * Instead: estimate noise floor from a strided sample of the power spectrum
 * (occupancy << 1% of 32768 bins), then apply a fixed threshold derived
 * analytically for the target Pfa.
 *
 * For complex AWGN, bin power follows Exp(λ), so:
 *   Threshold = λ · (−ln Pfa)         where λ = noise_floor (mean power/bin)
 *   30th percentile of Exp(λ) = −λ·ln(0.70) ≈ 0.357·λ
 *   → λ = perc30 / 0.357
 *   → Threshold = perc30 · (−ln Pfa) / 0.357
 *
 * For Pfa = 1e-4:  −ln(1e-4) ≈ 9.21  → Threshold ≈ perc30 × 25.8
 */
#include "wideband.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define NOISE_SAMPLE_N   1024      /* strided sample size for noise estimation */
#define NOISE_PERCENTILE 0.30f     /* 30th percentile                          */

static inline float q15_to_float(int16_t v) { return (float)v / 32768.0f; }

/* ── qsort comparator ────────────────────────────────────────────────── */
static int float_cmp(const void *a, const void *b)
{
    float fa = *(const float *)a;
    float fb = *(const float *)b;
    return (fa > fb) - (fa < fb);
}

/* ── Unpack DMA data → linear power[32768] ──────────────────────────── */
static void unpack_power(const WbDmaBin *dma, float *power)
{
    for (int local_bin = 0; local_bin < WB_BINS_PER_CH; local_bin++) {
        for (int ch = 0; ch < WB_N_CHANNELS; ch++) {
            float re = q15_to_float(dma[local_bin].i[ch]);
            float im = q15_to_float(dma[local_bin].q[ch]);
            power[ch * WB_BINS_PER_CH + local_bin] = re * re + im * im;
        }
    }
}

/* ── Noise floor estimation: strided sample + 30th percentile sort ──── */
static float _wb_sample[NOISE_SAMPLE_N];

static float estimate_noise_floor(const float *power)
{
    int stride = WB_TOTAL_BINS / NOISE_SAMPLE_N;
    for (int i = 0; i < NOISE_SAMPLE_N; i++)
        _wb_sample[i] = power[i * stride];

    qsort(_wb_sample, NOISE_SAMPLE_N, sizeof(float), float_cmp);

    int idx = (int)(NOISE_PERCENTILE * NOISE_SAMPLE_N);
    return _wb_sample[idx];                /* 30th percentile            */
}

/* ── Threshold from 30th percentile noise and target Pfa ────────────── */
static float noise_to_threshold(float perc30, float pfa)
{
    /* λ = perc30 / (-ln(0.70))   = perc30 / 0.3567            */
    /* T = λ · (-ln(pfa))                                       */
    float lambda = perc30 / (-logf(0.70f));
    return lambda * (-logf(pfa));
}

/* ── Power-weighted centroid ─────────────────────────────────────────── */
static float centroid_bin(const float *power, int start, int end)
{
    float sum_w = 0.0f, sum_wk = 0.0f;
    for (int k = start; k <= end; k++) {
        sum_w  += power[k];
        sum_wk += power[k] * (float)k;
    }
    return (sum_w > 0.0f) ? sum_wk / sum_w : (float)((start + end) / 2);
}

static float peak_power(const float *power, int start, int end, int *pk_bin)
{
    float best = power[start];
    *pk_bin = start;
    for (int k = start + 1; k <= end; k++) {
        if (power[k] > best) { best = power[k]; *pk_bin = k; }
    }
    return best;
}

/* ── Static work buffers ─────────────────────────────────────────────── */
static float _wb_power[WB_TOTAL_BINS];
static int   _wb_dets[WB_TOTAL_BINS];
static int   _wb_starts[DETECT_MAX_SIGNALS];
static int   _wb_ends[DETECT_MAX_SIGNALS];

/* ── Public API ──────────────────────────────────────────────────────── */

float wb_bin_to_freq(int global_bin, float rf_center_hz)
{
    float offset = ((float)global_bin - (float)(WB_TOTAL_BINS / 2))
                   * WB_FREQ_RES_HZ;
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
    float pfa     = params ? params->alpha : WB_CFAR_PFA;
    int   min_bins = WB_MIN_CLUSTER_BINS;

    /* Step 1: unpack DMA → power spectrum. */
    unpack_power(dma_buf, _wb_power);

    /* Step 2: estimate noise floor and compute threshold. */
    float noise_perc30 = estimate_noise_floor(_wb_power);
    float threshold    = noise_to_threshold(noise_perc30, pfa);

    /* Step 3: threshold detection. */
    int n_dets = 0;
    for (int k = 0; k < WB_TOTAL_BINS && n_dets < WB_TOTAL_BINS; k++) {
        if (_wb_power[k] > threshold)
            _wb_dets[n_dets++] = k;
    }

    /* Step 4: cluster adjacent detections (gap ≤ WB_CFAR_N_GUARD×2). */
    int gap     = WB_CFAR_N_GUARD * 2;
    int n_clust = cfar_cluster(_wb_dets, n_dets, gap,
                               _wb_starts, _wb_ends, DETECT_MAX_SIGNALS);

    /* Step 5: extract signal parameters per cluster. */
    out->n_signals = 0;
    for (int c = 0; c < n_clust && out->n_signals < DETECT_MAX_SIGNALS; c++) {
        int start = _wb_starts[c];
        int end   = _wb_ends[c];
        if (end - start + 1 < min_bins) continue;

        int   pk_bin;
        float pk_pwr = peak_power(_wb_power, start, end, &pk_bin);
        float cf_bin = centroid_bin(_wb_power, start, end);
        float cf_hz  = wb_bin_to_freq((int)(cf_bin + 0.5f), rf_center_hz);
        float bw_hz  = (float)(end - start + 1) * WB_FREQ_RES_HZ;
        float dbfs   = 10.0f * log10f(pk_pwr / 2.0f + 1e-30f);

        Signal *s = &out->signals[out->n_signals++];
        s->center_freq_hz = cf_hz;
        s->bandwidth_hz   = bw_hz;
        s->power_dbfs     = dbfs;
        s->wb_peak_bin    = pk_bin;
    }
}
