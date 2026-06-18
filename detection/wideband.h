/* detection/wideband.h — Wideband spectrum processing */
#ifndef WIDEBAND_H
#define WIDEBAND_H

#include "detection.h"
#include "cfar.h"

/* Default CFAR configuration for the wideband pass.
 * Freq resolution = 18.75 kHz → n_guard=4 (75 kHz), n_ref=16/side (300 kHz). */
#define WB_CFAR_N_GUARD   4
#define WB_CFAR_N_REF     16
#define WB_CFAR_PFA       1e-4f

/* Minimum cluster width (bins) below which a detection is discarded.
 * 1 bin = 18.75 kHz; 2-bin minimum → ~37.5 kHz minimum BW. */
#define WB_MIN_CLUSTER_BINS  2

/* Process a 128-KB wideband DMA buffer and return detected signals.
 *
 * dma_buf       : pointer to WB_DMA_BYTES of data (4096 × WbDmaBin)
 * rf_center_hz  : absolute frequency at the center of the 614.4 MHz span
 * params        : CFAR parameters (pass NULL to use defaults above)
 * out           : output signal list (caller-allocated)
 *
 * Fills out->signals[] and out->n_signals. */
void wb_detect(const WbDmaBin *dma_buf, float rf_center_hz,
               const CfarParams *params, SignalList *out);

/* Convert a wideband global bin index to an absolute frequency (Hz).
 * global_bin in [0, WB_TOTAL_BINS). */
float wb_bin_to_freq(int global_bin, float rf_center_hz);

/* Convert an absolute frequency to the nearest wideband global bin.
 * Returns -1 if out of range. */
int wb_freq_to_bin(float freq_hz, float rf_center_hz);

#endif /* WIDEBAND_H */
