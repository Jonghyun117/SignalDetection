/* tests/test_preprocess.c */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../inference/preprocess.h"

#define N 1024

static void fail(const char *msg, float got, float exp) {
    fprintf(stderr, "FAIL %s: expected %.6f, got %.6f\n", msg, exp, got);
    exit(1);
}

static void test_dc_removal(void) {
    float i_in[N], q_in[N], feat[3][N];
    /* Alternating signal with large DC offset */
    for (int t = 0; t < N; t++) {
        i_in[t] = 10.0f + (t % 2 == 0 ? 1.0f : -1.0f);
        q_in[t] = 5.0f  + (t % 2 == 0 ? 1.0f : -1.0f);
    }
    amc_preprocess(i_in, q_in, feat, N);
    /* After preprocessing, amplitude channel mean should reflect unit power, not DC */
    float mean_a = 0.0f;
    for (int t = 0; t < N; t++) mean_a += feat[0][t];
    mean_a /= N;
    if (fabsf(mean_a - 1.0f) > 0.01f)
        fail("dc_removal: mean amplitude", mean_a, 1.0f);
    printf("PASS: test_dc_removal\n");
}

static void test_power_normalization(void) {
    float i_in[N], q_in[N], feat[3][N];
    for (int t = 0; t < N; t++) {
        i_in[t] = (t % 2 == 0) ?  3.0f : -3.0f;
        q_in[t] = (t % 2 == 0) ?  4.0f : -4.0f;
    }
    amc_preprocess(i_in, q_in, feat, N);
    float mean_pow = 0.0f;
    for (int t = 0; t < N; t++) mean_pow += feat[0][t] * feat[0][t];
    mean_pow /= N;
    if (fabsf(mean_pow - 1.0f) > 0.02f)
        fail("power_normalization: mean(A^2)", mean_pow, 1.0f);
    printf("PASS: test_power_normalization\n");
}

static void test_bpsk_phase(void) {
    /* BPSK: alternating +/-1 on I, Q=0 -> phase alternates between 0 and +/-pi */
    float i_in[N], q_in[N], feat[3][N];
    for (int t = 0; t < N; t++) {
        i_in[t] = (t % 16 < 8) ? 1.0f : -1.0f;
        q_in[t] = 0.0f;
    }
    amc_preprocess(i_in, q_in, feat, N);
    for (int t = 0; t < N; t++) {
        float phi = feat[1][t];
        if (fabsf(phi) > 0.1f && fabsf(fabsf(phi) - 3.14159f) > 0.1f) {
            fprintf(stderr, "FAIL bpsk_phase: phi[%d]=%.4f\n", t, phi);
            exit(1);
        }
    }
    printf("PASS: test_bpsk_phase\n");
}

int main(void) {
    test_dc_removal();
    test_power_normalization();
    test_bpsk_phase();
    printf("All preprocess tests passed.\n");
    return 0;
}
