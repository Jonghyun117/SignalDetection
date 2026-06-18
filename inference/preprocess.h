/* inference/preprocess.h */
#ifndef AMC_PREPROCESS_H
#define AMC_PREPROCESS_H

/* Preprocess 2048-sample IQ into 3-channel instantaneous features.
   features[0]: amplitude |A(t)|
   features[1]: instantaneous phase phi(t)
   features[2]: instantaneous frequency delta_phi(t) (phase increment, unwrapped)
   n must equal 2048. */
void amc_preprocess(const float *i_samples, const float *q_samples,
                    float features[3][2048], int n);

#endif
