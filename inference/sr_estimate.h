/* inference/sr_estimate.h
 * Symbol rate estimation from baseband IQ after AMC classification.
 */
#ifndef SR_ESTIMATE_H
#define SR_ESTIMATE_H

/* Estimate symbol rate using the AMC result to select the right method.
 *
 * i_in, q_in  : raw baseband IQ at current (unknown) sampling rate [1024]
 * class_idx   : AMC result from amc_forward()  [0 .. AMC_NUM_CLASSES-1]
 *
 * Returns  f_sym / f_s  (normalized, range 0 < result <= 0.5)
 *   → hardware symbol rate  = result × f_s_hardware_hz
 *   → set next sample rate  = 4 × hardware symbol rate
 *
 * Returns 0.0f for CW / FM (no symbol rate) or when SNR is too low
 * to find a reliable spectral peak.
 */
float amc_sr_estimate(const float i_in[1024], const float q_in[1024],
                      int class_idx);

#endif /* SR_ESTIMATE_H */
