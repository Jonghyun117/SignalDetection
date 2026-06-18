/* detection/main_detect.c
 *
 * Signal detection pipeline demonstration / integration test.
 *
 * Usage:
 *   ./detect <scenario.json> <wideband.bin> [narrowband.bin]
 *
 * wideband.bin  : raw 128 KB DMA dump (4096 × WbDmaBin)
 * narrowband.bin: optional raw 8 KB NB DMA dump per signal (use first found)
 */
#include "scenario.h"
#include "wideband.h"
#include "narrowband.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int read_file(const char *path, void *buf, size_t expected)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open: %s\n", path); return -1; }
    size_t got = fread(buf, 1, expected, f);
    fclose(f);
    if (got != expected) {
        fprintf(stderr, "%s: expected %zu bytes, got %zu\n", path, expected, got);
        return -1;
    }
    return 0;
}

static WbDmaBin   _wb_buf[WB_BINS_PER_CH];   /* 128 KB                  */
static NbDmaBlock _nb_buf;                     /*   8 KB                  */

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <scenario.json> <wb.bin> [nb.bin]\n", argv[0]);
        return 1;
    }

    /* ── 1. Load scenario ─────────────────────────────────────────────── */
    Scenario sc;
    if (scenario_load(argv[1], &sc) != 0) {
        fprintf(stderr, "Failed to load scenario: %s\n", argv[1]);
        return 1;
    }
    scenario_print(&sc);

    /* ── 2. Wideband detection ────────────────────────────────────────── */
    if (read_file(argv[2], _wb_buf, WB_DMA_BYTES) != 0) return 1;

    SignalList wb_signals;
    wb_detect(_wb_buf, (float)sc.rf_center_hz, NULL, &wb_signals);

    printf("\n── Wideband Detection (span=%.1f MHz, res=%.3f kHz) ──\n",
           WB_TOTAL_BW_HZ / 1e6f, WB_FREQ_RES_HZ / 1e3f);
    printf("  %d signal(s) found\n", wb_signals.n_signals);

    for (int i = 0; i < wb_signals.n_signals; i++) {
        Signal *s = &wb_signals.signals[i];
        printf("  [%d] CF=%.3f MHz  BW=%.1f kHz  Power=%.1f dBFS\n", i,
               s->center_freq_hz / 1e6f,
               s->bandwidth_hz   / 1e3f,
               s->power_dbfs);
    }

    /* ── 3. Narrowband refinement (optional, first signal) ───────────── */
    if (argc >= 4 && wb_signals.n_signals > 0) {
        if (read_file(argv[3], &_nb_buf, NB_DMA_BYTES) == 0) {
            /* RF would be tuned to wb_signals.signals[0].center_freq_hz
             * before capturing the NB block. Pass that as nb_rf_center. */
            float nb_center = wb_signals.signals[0].center_freq_hz;
            Signal nb_sig;
            int found = nb_detect(&_nb_buf, nb_center, NULL, &nb_sig);

            printf("\n── Narrowband Refinement (span=%.3f MHz, res=%.2f Hz) ──\n",
                   NB_BW_HZ / 1e6f, NB_FREQ_RES_HZ);
            if (found) {
                printf("  CF=%.6f MHz  BW=%.3f kHz  Power=%.1f dBFS\n",
                       nb_sig.center_freq_hz / 1e6f,
                       nb_sig.bandwidth_hz   / 1e3f,
                       nb_sig.power_dbfs);
            } else {
                printf("  No signal detected in narrowband.\n");
            }
        }
    }

    return 0;
}
