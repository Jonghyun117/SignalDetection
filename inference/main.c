/* inference/main.c
 * AMC inference entry point — pure C, no external dependencies.
 *
 * Input  (stdin) : float32 I[2048] then float32 Q[2048], binary, repeated per frame.
 * Output (stdout): "<class> conf=X.XXX\n" per frame
 *
 * Vitis 사용법:
 *   preprocess.c, nn_forward.c, main.c 를 프로젝트에 추가
 *   nn_weights.h 는 PC에서 먼저 생성: python training/export_c.py
 */
#include <stdio.h>
#include "preprocess.h"
#include "nn_forward.h"
#include "nn_weights.h"

#define N 2048

int main(void)
{
    float i_buf[N], q_buf[N];
    float features[3][N];
    float probs[AMC_NUM_CLASSES];

    while (fread(i_buf, sizeof(float), N, stdin) == (size_t)N &&
           fread(q_buf, sizeof(float), N, stdin) == (size_t)N)
    {
        amc_preprocess(i_buf, q_buf, features, N);
        int pred = amc_forward(features, probs);
        printf("%s conf=%.3f\n", AMC_CLASS_NAMES[pred], probs[pred]);
        fflush(stdout);
    }

    return 0;
}
