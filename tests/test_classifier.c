/* tests/test_classifier.c */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../inference/classifier.h"

#ifndef MODEL_PATH
#define MODEL_PATH "model/amc_model.onnx"
#endif

static void test_init_and_destroy(void) {
    if (amc_classifier_init(MODEL_PATH) != 0) {
        fprintf(stderr, "FAIL test_init_and_destroy\n"); exit(1);
    }
    amc_classifier_destroy();
    printf("PASS: test_init_and_destroy\n");
}

static void test_output_is_probability(void) {
    amc_classifier_init(MODEL_PATH);

    float features[3][2048] = {{0}};
    for (int t = 0; t < 2048; t++) features[0][t] = 1.0f;

    float probs[AMC_NUM_CLASSES];
    int pred = amc_classifier_run(features, probs);

    if (pred < 0 || pred >= AMC_NUM_CLASSES) {
        fprintf(stderr, "FAIL: pred=%d out of range\n", pred); exit(1);
    }
    float sum = 0.0f;
    for (int i = 0; i < AMC_NUM_CLASSES; i++) sum += probs[i];
    if (fabsf(sum - 1.0f) > 0.01f) {
        fprintf(stderr, "FAIL: probs sum=%.4f\n", sum); exit(1);
    }
    printf("PASS: test_output_is_probability (pred=%s, conf=%.3f)\n",
           AMC_CLASS_NAMES[pred], probs[pred]);
    amc_classifier_destroy();
}

int main(void) {
    test_init_and_destroy();
    test_output_is_probability();
    printf("All classifier tests passed.\n");
    return 0;
}
