/* detection/narrowband.h — Narrowband spectrum processing */
#ifndef NARROWBAND_H
#define NARROWBAND_H

#include "detection.h"
#include "cfar.h"

/* Default CFAR for narrowband pass.
 * Freq res = 937.5 Hz → n_guard=4 (3.75 kHz), n_ref=8/side (7.5 kHz). */
#define NB_CFAR_N_GUARD   4
#define NB_CFAR_N_REF     8
#define NB_CFAR_PFA       1e-4f

/* Process a narrowband DMA block and return refined signal parameters.
 *
 * dma_buf      : pointer to NbDmaBlock (8 KB, I-block then Q-block)
 * rf_center_hz : RF center frequency this capture was tuned to
 * params       : CFAR params (NULL = use defaults)
 * out          : output Signal (caller-allocated); populated on success
 *
 * Returns 1 if at least one signal detected, 0 otherwise.
 * When multiple clusters are found, the strongest is returned in out. */
int nb_detect(const NbDmaBlock *dma_buf, float rf_center_hz,
              const CfarParams *params, Signal *out);

/* Convert a narrowband bin index to absolute frequency (Hz). */
float nb_bin_to_freq(int bin, float rf_center_hz);

#endif /* NARROWBAND_H */
