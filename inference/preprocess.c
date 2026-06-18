/* inference/preprocess.c */
#include "preprocess.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void amc_preprocess(const float *i_in, const float *q_in,
                    float features[3][2048], int n)
{
    float i_buf[2048], q_buf[2048], phi[2048];
    int t;

    /* Channel 0: raw amplitude BEFORE DC removal.
       Preserves OOK on/off envelope and AM sinusoidal envelope. */
    float amp_pwr = 0.0f;
    for (t = 0; t < n; t++) {
        float a = sqrtf(i_in[t]*i_in[t] + q_in[t]*q_in[t]);
        features[0][t] = a;
        amp_pwr += a * a;
    }
    amp_pwr /= n;
    float amp_scale = (amp_pwr > 1e-10f) ? 1.0f / sqrtf(amp_pwr) : 1.0f;
    for (t = 0; t < n; t++) features[0][t] *= amp_scale;

    /* DC removal */
    float mi = 0.0f, mq = 0.0f;
    for (t = 0; t < n; t++) { mi += i_in[t]; mq += q_in[t]; }
    mi /= n; mq /= n;
    for (t = 0; t < n; t++) { i_buf[t] = i_in[t] - mi; q_buf[t] = q_in[t] - mq; }

    /* Power normalization: mean(I^2+Q^2) = 1 */
    float power = 0.0f;
    for (t = 0; t < n; t++) power += i_buf[t]*i_buf[t] + q_buf[t]*q_buf[t];
    power /= n;
    float scale = (power > 1e-10f) ? 1.0f / sqrtf(power) : 1.0f;
    for (t = 0; t < n; t++) { i_buf[t] *= scale; q_buf[t] *= scale; }

    /* Channel 1: instantaneous phase (on DC-removed, normalized signal) */
    for (t = 0; t < n; t++) {
        phi[t]         = atan2f(q_buf[t], i_buf[t]);
        features[1][t] = phi[t];
    }

    /* Channel 2: instantaneous frequency (unwrapped phase diff) */
    features[2][0] = 0.0f;
    for (t = 1; t < n; t++) {
        float d = phi[t] - phi[t-1];
        while (d >  (float)M_PI) d -= 2.0f * (float)M_PI;
        while (d < -(float)M_PI) d += 2.0f * (float)M_PI;
        features[2][t] = d;
    }
}
