/* inference/main.c
   Usage: ./amc_infer <model.onnx>
   Input  (stdin) : float32 I[1024] then float32 Q[1024], binary, repeated per frame.
   Output (stdout): "<class> conf=X.XXX latency=XX.XXms\n" per frame. */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "preprocess.h"
#include "classifier.h"

#define N 1024

static double elapsed_ms(struct timespec *t0, struct timespec *t1) {
    return (t1->tv_sec - t0->tv_sec) * 1e3 + (t1->tv_nsec - t0->tv_nsec) * 1e-6;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.onnx>\n", argv[0]);
        return 1;
    }
    if (amc_classifier_init(argv[1]) != 0) {
        fprintf(stderr, "Classifier init failed\n");
        return 1;
    }

    float i_buf[N], q_buf[N], features[3][N], probs[AMC_NUM_CLASSES];

    while (fread(i_buf, sizeof(float), N, stdin) == (size_t)N &&
           fread(q_buf, sizeof(float), N, stdin) == (size_t)N)
    {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        amc_preprocess(i_buf, q_buf, features, N);
        int pred = amc_classifier_run(features, probs);

        clock_gettime(CLOCK_MONOTONIC, &t1);

        if (pred >= 0)
            printf("%s conf=%.3f latency=%.2fms\n",
                   AMC_CLASS_NAMES[pred], probs[pred], elapsed_ms(&t0, &t1));
    }

    amc_classifier_destroy();
    return 0;
}
