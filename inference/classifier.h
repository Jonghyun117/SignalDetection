/* inference/classifier.h */
#ifndef AMC_CLASSIFIER_H
#define AMC_CLASSIFIER_H

#define AMC_NUM_CLASSES 21

/* Class names in MODULATIONS order (matches training/simulate.py). */
extern const char *AMC_CLASS_NAMES[AMC_NUM_CLASSES];

/* Load model. Returns 0 on success, -1 on failure. */
int  amc_classifier_init(const char *model_path);

/* Run inference. features[3][2048]: preprocessed input.
   probs[21]: softmax output. Returns predicted class index, or -1 on error. */
int  amc_classifier_run(float features[3][2048], float probs[AMC_NUM_CLASSES]);

/* Release ONNX Runtime resources. */
void amc_classifier_destroy(void);

#endif
