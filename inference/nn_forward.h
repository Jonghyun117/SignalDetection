/* inference/nn_forward.h
 * Pure-C AMCNet forward pass — no external dependencies.
 * Include nn_weights.h before this header (or let nn_forward.c include it).
 */
#ifndef NN_FORWARD_H
#define NN_FORWARD_H

/* Top-level inference call.
 * features : float[3][2048] from amc_preprocess()
 * probs_out : float[AMC_NUM_CLASSES] — softmax output (may be NULL)
 * returns   : predicted class index [0, AMC_NUM_CLASSES)
 */
int amc_forward(const float features[3][2048], float *probs_out);

#endif /* NN_FORWARD_H */
