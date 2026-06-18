/* detection/detection.h
 * Common types and constants for the signal detection pipeline.
 *
 * Pipeline:
 *   scenario.json → RF center → WB DMA (128 KB) → CFAR → Signal list
 *   → per signal: RF tune → NB DMA (8 KB) → CFAR → refined params → AMC
 */
#ifndef DETECTION_H
#define DETECTION_H

#include <stdint.h>

/* ── Wideband constants ────────────────────────────────────────────────── */
#define WB_N_CHANNELS       8
#define WB_BINS_PER_CH      4096
#define WB_TOTAL_BINS       (WB_N_CHANNELS * WB_BINS_PER_CH)   /* 32768   */
#define WB_FS_PER_CH_HZ     76800000.0f     /* 76.8 MHz per channel        */
#define WB_TOTAL_BW_HZ      614400000.0f    /* 614.4 MHz                   */
#define WB_FREQ_RES_HZ      18750.0f        /* 76.8 MHz / 4096             */
#define WB_DMA_BYTES        131072          /* 4096 * 32 = 128 KB          */

/* ── Narrowband constants ─────────────────────────────────────────────── */
#define NB_N_POINTS         2048
#define NB_FREQ_RES_HZ      937.5f
#define NB_BW_HZ            1920000.0f      /* 2048 * 937.5 Hz = 1.92 MHz  */
#define NB_DMA_BYTES        8192            /* 2048 * 4 (I 2B + Q 2B)      */

/* ── Result limits ────────────────────────────────────────────────────── */
#define DETECT_MAX_SIGNALS  64

/* ── DMA data types ───────────────────────────────────────────────────── */

/* One 32-byte DMA bin: I and Q for all 8 channels at the same FFT index.
 * The 8 channels are laid out sequentially in frequency:
 *   global_bin = ch * WB_BINS_PER_CH + local_bin
 * Channel 0 covers the lowest 76.8 MHz, channel 7 the highest. */
typedef struct {
    int16_t i[WB_N_CHANNELS];   /* Q15: real parts of 8 ch FFT output     */
    int16_t q[WB_N_CHANNELS];   /* Q15: imag parts of 8 ch FFT output     */
} WbDmaBin;                     /* sizeof = 32 bytes                       */

/* Narrowband single-channel complex FFT output: I-block then Q-block.    */
typedef struct {
    int16_t i[NB_N_POINTS];     /* Q15 real                               */
    int16_t q[NB_N_POINTS];     /* Q15 imag                               */
} NbDmaBlock;                   /* sizeof = 8192 bytes                     */

/* ── Signal descriptor ────────────────────────────────────────────────── */
typedef struct {
    float   center_freq_hz;     /* absolute center frequency               */
    float   bandwidth_hz;       /* estimated occupied bandwidth            */
    float   power_dbfs;         /* peak power relative to full-scale (dBFS)*/
    int     wb_peak_bin;        /* wideband global bin index of peak       */
} Signal;

typedef struct {
    Signal  signals[DETECT_MAX_SIGNALS];
    int     n_signals;
} SignalList;

#endif /* DETECTION_H */
