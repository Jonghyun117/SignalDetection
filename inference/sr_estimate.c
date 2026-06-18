/* inference/sr_estimate.c
 *
 * Symbol rate estimation from 2048 baseband IQ samples.
 *
 * Three methods selected by AMC class:
 *
 *   Method A  |r|² power spectrum
 *     For PSK / QAM / APSK / AM / PM.
 *     RRC pulse shaping creates amplitude ripple at every symbol boundary.
 *     FFT( |r[n]|² − mean ) → spectral line at f_sym.
 *
 *   Method A'  |r|² power spectrum with Nyquist-extended search (OQPSK)
 *     OQPSK Q-branch is delayed by SPS/2 samples.
 *     In |r|², I² and Q² spectral components at f_sym cancel (phase diff = π).
 *     The dominant spectral line appears at 2×f_sym.
 *     Search extended to include Nyquist bin; result is halved to yield f_sym.
 *
 *   Method B  |Δdphi| phase-acceleration spectrum
 *     For FSK (CPFSK has constant envelope, Method A fails).
 *     Instantaneous frequency dphi[n] = arg(z[n]·z*[n−1]).
 *     Between symbols: dphi ≈ const → Δdphi ≈ 0.
 *     At symbol transition: dphi jumps → |Δdphi| spikes.
 *     FFT( |dphi[n]−dphi[n−1]| − mean ) → spectral line at f_sym.
 *
 * Peak quality gate: if peak/mean_spectrum < QUALITY_THR the signal is
 * too noisy and 0.0f is returned.
 */
#include "sr_estimate.h"
#include <math.h>

#define N            2048
#define N_2          1024
#define QUALITY_THR  3.0f   /* peak must be >3× mean spectral power */
#define PI           3.14159265358979f
#define TWO_PI       6.28318530717959f

/* ── static work buffers ─────────────────────────────────────────────── */
static float _re[N], _im[N];

/* ── radix-2 in-place FFT for any power-of-2 N ──────────────────────── */
static void fft_radix2(float *re, float *im)
{
    /* Bit-reversal permutation (log2(N)-bit reverse, works for any power-of-2 N) */
    for (int i = 1, j = 0; i < N; i++) {
        int bit = N >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float t;
            t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
    /* Cooley-Tukey butterfly */
    for (int len = 2; len <= N; len <<= 1) {
        float ang = -TWO_PI / (float)len;
        float c0  = cosf(ang), s0 = sinf(ang);
        for (int i = 0; i < N; i += len) {
            float wr = 1.0f, wi = 0.0f;
            int   half = len >> 1;
            for (int k = 0; k < half; k++) {
                float ur = re[i+k],    ui = im[i+k];
                float vr = re[i+k+half]*wr - im[i+k+half]*wi;
                float vi = re[i+k+half]*wi + im[i+k+half]*wr;
                re[i+k]      = ur + vr;  im[i+k]      = ui + vi;
                re[i+k+half] = ur - vr;  im[i+k+half] = ui - vi;
                float nwr = wr*c0 - wi*s0;
                wi = wr*s0 + wi*c0;
                wr = nwr;
            }
        }
    }
}

/* Hann window (reduces spectral leakage) */
static void hann_window(float *x)
{
    for (int i = 0; i < N; i++)
        x[i] *= 0.5f * (1.0f - cosf(TWO_PI * i / (N - 1)));
}

/* Find peak bin in |FFT|² over k=1..end (inclusive).
 * Also computes mean of remaining bins for quality check.
 * Returns best_k; sets *quality = peak_power / mean_power. */
static int peak_bin_range(int end, float *quality_out)
{
    int   best_k   = 1;
    float best_mag = _re[1]*_re[1] + _im[1]*_im[1];
    float sum_mag  = best_mag;

    for (int k = 2; k <= end; k++) {
        float m = _re[k]*_re[k] + _im[k]*_im[k];
        sum_mag += m;
        if (m > best_mag) { best_mag = m; best_k = k; }
    }

    float mean_mag  = (sum_mag - best_mag) / (float)(end - 1);
    *quality_out    = (mean_mag > 0.0f) ? best_mag / mean_mag : 0.0f;
    return best_k;
}

/* ── Method A: power spectrum (PSK / QAM / APSK) ────────────────────── */
static float estimate_power(const float *I, const float *Q)
{
    float mean = 0.0f;
    for (int n = 0; n < N; n++) {
        _re[n] = I[n]*I[n] + Q[n]*Q[n];
        _im[n] = 0.0f;
        mean  += _re[n];
    }
    mean /= N;
    for (int n = 0; n < N; n++) _re[n] -= mean;

    hann_window(_re);
    fft_radix2(_re, _im);

    float quality;
    int k = peak_bin_range(N_2 - 1, &quality);
    return (quality >= QUALITY_THR) ? (float)k / N : 0.0f;
}

/* ── Method A': I/Q-separated power spectrum for OQPSK ──────────────── */
/* OQPSK Q-branch delayed by SPS/2 → in |I|²+|Q|², the f_sym spectral
 * components cancel (phase diff = π at f_sym · T_sym/2 = π/2, multiplied
 * by 2 from the squaring: net phase = π → destructive).
 * Using the incoherent sum |FFT(I²)|² + |FFT(Q²)|² avoids cancellation:
 * both channels individually peak at f_sym, and magnitudes add. */
static float estimate_power_oqpsk(const float *I, const float *Q)
{
    float mean_i = 0.0f, mean_q = 0.0f;
    for (int n = 0; n < N; n++) {
        _re[n]  = I[n]*I[n];
        mean_i += _re[n];
    }
    mean_i /= N;
    for (int n = 0; n < N; n++) { _re[n] -= mean_i; _im[n] = 0.0f; }
    hann_window(_re);
    fft_radix2(_re, _im);

    /* Store |FFT(I²)|² in a temporary array using the second half of _im */
    /* Reuse _im as temp: store real^2+imag^2 in first N_2 slots */
    static float _tmp[N_2];
    for (int k = 0; k < N_2; k++) _tmp[k] = _re[k]*_re[k] + _im[k]*_im[k];

    /* Compute FFT(Q²) */
    for (int n = 0; n < N; n++) {
        _re[n]  = Q[n]*Q[n];
        mean_q += _re[n];
    }
    mean_q /= N;
    for (int n = 0; n < N; n++) { _re[n] -= mean_q; _im[n] = 0.0f; }
    hann_window(_re);
    fft_radix2(_re, _im);

    /* Incoherent sum: |FFT(I²)|² + |FFT(Q²)|² */
    int   best_k   = 1;
    float best_sum = _tmp[1] + _re[1]*_re[1] + _im[1]*_im[1];
    float sum_all  = best_sum;
    for (int k = 2; k < N_2; k++) {
        float m = _tmp[k] + _re[k]*_re[k] + _im[k]*_im[k];
        sum_all += m;
        if (m > best_sum) { best_sum = m; best_k = k; }
    }
    float mean_rest = (sum_all - best_sum) / (float)(N_2 - 2);
    float quality   = (mean_rest > 0.0f) ? best_sum / mean_rest : 0.0f;
    return (quality >= QUALITY_THR) ? (float)best_k / N : 0.0f;
}

/* ── Method C: envelope spectrum (OOK) ──────────────────────────────── */
static float estimate_envelope(const float *I, const float *Q)
{
    float mean = 0.0f;
    for (int n = 0; n < N; n++) {
        _re[n] = sqrtf(I[n]*I[n] + Q[n]*Q[n]);
        _im[n] = 0.0f;
        mean  += _re[n];
    }
    mean /= N;
    for (int n = 0; n < N; n++) _re[n] -= mean;

    hann_window(_re);
    fft_radix2(_re, _im);

    float quality;
    int k = peak_bin_range(N_2 - 1, &quality);
    return (quality >= QUALITY_THR) ? (float)k / N : 0.0f;
}

/* ── Method B: phase-acceleration spectrum (FSK) ─────────────────────── */
static float estimate_fsk(const float *I, const float *Q)
{
    float prev_dphi = 0.0f, mean = 0.0f;

    for (int n = 1; n < N; n++) {
        float cr  =  I[n]*I[n-1] + Q[n]*Q[n-1];
        float ci  =  Q[n]*I[n-1] - I[n]*Q[n-1];
        float dphi = atan2f(ci, cr);

        float dd = dphi - prev_dphi;
        if (dd >  PI) dd -= TWO_PI;
        if (dd < -PI) dd += TWO_PI;

        _re[n-1]  = (dd < 0.0f) ? -dd : dd;
        mean     += _re[n-1];
        prev_dphi = dphi;
    }
    _re[N-1] = 0.0f;
    _im[0]   = 0.0f;

    mean /= (N - 1);
    for (int n = 0; n < N; n++) { _re[n] -= mean; _im[n] = 0.0f; }

    hann_window(_re);
    fft_radix2(_re, _im);

    float quality;
    int k = peak_bin_range(N_2 - 1, &quality);
    return (quality >= QUALITY_THR) ? (float)k / N : 0.0f;
}

/* ── Public API ──────────────────────────────────────────────────────── */

/*
 * CLASSES (from nn_weights.h):
 *   0=AM  1=FM  2=PM  3=CW  4=BPSK/DBPSK  5=QPSK/DQPSK  6=8PSK
 *   7=OQPSK  8=2FSK  9=4FSK  10=8FSK  11=8QAM  12=16QAM
 *   13=16APSK 14=32APSK 15=64APSK 16=128APSK 17=256APSK  18=OOK
 */
float amc_sr_estimate(const float i_in[2048], const float q_in[2048],
                      int class_idx)
{
    /* Analog / CW / OOK: symbol rate not estimable. */
    if (class_idx == 0 || class_idx == 1 || class_idx == 2 ||
        class_idx == 3 || class_idx == 18)
        return 0.0f;

    /* FSK: phase-acceleration method (CPFSK has constant envelope) */
    if (class_idx == 8 || class_idx == 9 || class_idx == 10)
        return estimate_fsk(i_in, q_in);

    /* OQPSK: power spectrum with Nyquist-extended search + halved result */
    if (class_idx == 7)
        return estimate_power_oqpsk(i_in, q_in);

    /* PSK / QAM / APSK: standard power spectrum method */
    return estimate_power(i_in, q_in);
}
