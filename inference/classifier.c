/* inference/classifier.c */
#include "classifier.h"
#include <stdio.h>
#include <string.h>
#include "onnxruntime_c_api.h"

const char *AMC_CLASS_NAMES[AMC_NUM_CLASSES] = {
    "AM","FM","PM","CW",
    "2PSK","4PSK","8PSK",
    "DBPSK","DQPSK","OQPSK",
    "2FSK","4FSK","8FSK",
    "8QAM","16QAM",
    "16APSK","32APSK","64APSK","128APSK","256APSK",
    "OOK"
};

static const OrtApi    *g_ort  = NULL;
static OrtEnv          *g_env  = NULL;
static OrtSession      *g_sess = NULL;
static OrtMemoryInfo   *g_mem  = NULL;

int amc_classifier_init(const char *model_path)
{
    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!g_ort) return -1;

    OrtStatus *st;
    if ((st = g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "amc", &g_env))) {
        fprintf(stderr, "ORT CreateEnv: %s\n", g_ort->GetErrorMessage(st)); return -1;
    }

    OrtSessionOptions *opts;
    g_ort->CreateSessionOptions(&opts);
    g_ort->SetIntraOpNumThreads(opts, 1);
    g_ort->SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_ALL);

    if ((st = g_ort->CreateSession(g_env, model_path, opts, &g_sess))) {
        fprintf(stderr, "ORT CreateSession: %s\n", g_ort->GetErrorMessage(st));
        g_ort->ReleaseSessionOptions(opts); return -1;
    }
    g_ort->ReleaseSessionOptions(opts);
    g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &g_mem);
    return 0;
}

int amc_classifier_run(float features[3][1024], float probs[AMC_NUM_CLASSES])
{
    int64_t    shape[] = {1, 3, 1024};
    OrtValue  *in_val  = NULL, *out_val = NULL;
    OrtStatus *st;

    st = g_ort->CreateTensorWithDataAsOrtValue(
            g_mem, features, 3*1024*sizeof(float),
            shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in_val);
    if (st) return -1;

    const char *in_names[]  = {"input"};
    const char *out_names[] = {"output"};
    st = g_ort->Run(g_sess, NULL,
                    in_names,  (const OrtValue *const *)&in_val,  1,
                    out_names, 1, &out_val);
    g_ort->ReleaseValue(in_val);
    if (st) { fprintf(stderr, "ORT Run: %s\n", g_ort->GetErrorMessage(st)); return -1; }

    float *data;
    g_ort->GetTensorMutableData(out_val, (void **)&data);
    memcpy(probs, data, AMC_NUM_CLASSES * sizeof(float));
    g_ort->ReleaseValue(out_val);

    int best = 0;
    for (int i = 1; i < AMC_NUM_CLASSES; i++)
        if (probs[i] > probs[best]) best = i;
    return best;
}

void amc_classifier_destroy(void)
{
    if (g_mem)  { g_ort->ReleaseMemoryInfo(g_mem);  g_mem  = NULL; }
    if (g_sess) { g_ort->ReleaseSession(g_sess);     g_sess = NULL; }
    if (g_env)  { g_ort->ReleaseEnv(g_env);          g_env  = NULL; }
}
