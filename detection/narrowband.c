/* detection/narrowband.c — Narrowband spectrum processing
 *
 * Same percentile-based noise floor approach as wideband, sized for 2048 bins.
 * After RF tuning to the wideband-detected center, the primary signal occupies
 * up to ~NB_BW/2 = 960 kHz. Using CA-CFAR here would require n_guard ≈ 512
 * for very wide signals, so we use the same global threshold method.
 *
 * For refined BW estimation, the cluster extent × freq_res gives a 937.5 Hz
 * resolution estimate (vs. 18.75 kHz from wideband).
 */
#include "narrowband.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define NB_NOISE_SAMPLE_N   256
#define NB_NOISE_PERCENTILE 0.30f

static inline float q15_to_float(int16_t v) { return (float)v / 32768.0f; }

static int float_cmp(const void *a, const void *b) {
    float fa = *(const float *)a, fb = *(const float *)b;
    return (fa > fb) - (fa < fb);
}

static float _nb_power[NB_N_POINTS];
static float _nb_sample[NB_NOISE_SAMPLE_N];
static int   _nb_dets[NB_N_POINTS];
static int   _nb_starts[DETECT_MAX_SIGNALS];
static int   _nb_ends[DETECT_MAX_SIGNALS];

static float nb_noise_floor(void)
{
    int stride = NB_N_POINTS / NB_NOISE_SAMPLE_N;
    for (int i = 0; i < NB_NOISE_SAMPLE_N; i++)
        _nb_sample[i] = _nb_power[i * stride];
    qsort(_nb_sample, NB_NOISE_SAMPLE_N, sizeof(float), float_cmp);
    int idx = (int)(NB_NOISE_PERCENTILE * NB_NOISE_SAMPLE_N);
    return _nb_sample[idx];
}

static float noise_to_threshold(float perc30, float pfa)
{
    float lambda = perc30 / (-logf(0.70f));
    return lambda * (-logf(pfa));
}

float nb_bin_to_freq(int bin, float rf_center_hz)
{
    float offset = ((float)bin - (float)(NB_N_POINTS / 2)) * NB_FREQ_RES_HZ;
    return rf_center_hz + offset;
}

int nb_detect(const NbDmaBlock *dma_buf, float rf_center_hz,
              const CfarParams *params, Signal *out)
{
    float pfa = params ? params->alpha : NB_CFAR_PFA;

    for (int k = 0; k < NB_N_POINTS; k++) {
        float re = q15_to_float(dma_buf->iq[k * 2]);
        float im = q15_to_float(dma_buf->iq[k * 2 + 1]);
        _nb_power[k] = re * re + im * im;
    }

    float threshold = noise_to_threshold(nb_noise_floor(), pfa);

    int n_dets = 0;
    for (int k = 0; k < NB_N_POINTS && n_dets < NB_N_POINTS; k++) {
        if (_nb_power[k] > threshold)
            _nb_dets[n_dets++] = k;
    }

    int gap     = NB_CFAR_N_GUARD * 2;
    int n_clust = cfar_cluster(_nb_dets, n_dets, gap,
                               _nb_starts, _nb_ends, DETECT_MAX_SIGNALS);
    if (n_clust == 0) return 0;

    /* Select strongest cluster. */
    int best_c = 0;
    float best_pk = 0.0f;
    for (int c = 0; c < n_clust; c++) {
        for (int k = _nb_starts[c]; k <= _nb_ends[c]; k++) {
            if (_nb_power[k] > best_pk) { best_pk = _nb_power[k]; best_c = c; }
        }
    }

    int start = _nb_starts[best_c];
    int end   = _nb_ends[best_c];
    float sum_w = 0.0f, sum_wk = 0.0f, peak = 0.0f;
    int pk_bin = start;
    for (int k = start; k <= end; k++) {
        float p = _nb_power[k];
        sum_w  += p;
        sum_wk += p * (float)k;
        if (p > peak) { peak = p; pk_bin = k; }
    }
    float cf_bin = (sum_w > 0.0f) ? sum_wk / sum_w : (float)((start + end) / 2);

    out->center_freq_hz = nb_bin_to_freq((int)(cf_bin + 0.5f), rf_center_hz);
    out->bandwidth_hz   = (float)(end - start + 1) * NB_FREQ_RES_HZ;
    out->power_dbfs     = 10.0f * log10f(peak / 2.0f + 1e-30f);
    out->wb_peak_bin    = pk_bin;
    return 1;
}
