/* detection/narrowband.c — Narrowband spectrum processing
 *
 * Input: NbDmaBlock — int16 Q15 interleaved [I0,Q0,I1,Q1,...]
 * Noise floor: 30th percentile of strided samples (robust to wide signals).
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
    return _nb_sample[(int)(NB_NOISE_PERCENTILE * NB_NOISE_SAMPLE_N)];
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
    for (int k = 0; k < NB_N_POINTS && n_dets < NB_N_POINTS; k++)
        if (_nb_power[k] > threshold)
            _nb_dets[n_dets++] = k;

    int gap     = NB_CFAR_N_GUARD * 2;
    int n_clust = cfar_cluster(_nb_dets, n_dets, gap,
                               _nb_starts, _nb_ends, DETECT_MAX_SIGNALS);
    if (n_clust == 0) return 0;

    int best_c = 0; float best_pk = 0.0f;
    for (int c = 0; c < n_clust; c++)
        for (int k = _nb_starts[c]; k <= _nb_ends[c]; k++)
            if (_nb_power[k] > best_pk) { best_pk = _nb_power[k]; best_c = c; }

    int start = _nb_starts[best_c], end = _nb_ends[best_c];
    float sw = 0.0f, swk = 0.0f, peak = 0.0f; int pk_bin = start;
    for (int k = start; k <= end; k++) {
        float p = _nb_power[k];
        sw += p; swk += p * k;
        if (p > peak) { peak = p; pk_bin = k; }
    }
    float cf_bin = (sw > 0.0f) ? swk / sw : (float)((start + end) / 2);

    out->center_freq_hz = nb_bin_to_freq((int)(cf_bin + 0.5f), rf_center_hz);
    out->bandwidth_hz   = (float)(end - start + 1) * NB_FREQ_RES_HZ;
    out->power_dbfs     = 10.0f * log10f(peak / 2.0f + 1e-30f);
    out->wb_peak_bin    = pk_bin;
    return 1;
}
