#!/usr/bin/env python3
"""
detection/sim_detect.py
Simulate the wideband + narrowband detection pipeline in Python.

Detection strategy:
  - Noise floor: 30th percentile of strided samples (robust against masking)
  - Threshold:   noise_floor / (-ln 0.70) * (-ln Pfa)   [Exp(lambda) model]
  - Clustering:  adjacent bins merged with configurable gap

Usage:
    python detection/sim_detect.py [--rf_center_mhz 3500] [--out_dir detection/testdata]
"""
import argparse
import json
import os
import struct
import numpy as np

# ── Constants (mirror detection.h) ────────────────────────────────────────
WB_N_CH        = 8
WB_BINS_PER_CH = 4096
WB_TOTAL_BINS  = WB_N_CH * WB_BINS_PER_CH   # 32768
WB_FS_HZ       = 76.8e6
WB_FREQ_RES    = WB_FS_HZ / WB_BINS_PER_CH  # 18750 Hz
WB_TOTAL_BW    = WB_TOTAL_BINS * WB_FREQ_RES  # 614.4 MHz

NB_N_POINTS    = 2048
NB_FREQ_RES    = 937.5                        # Hz
NB_BW          = NB_N_POINTS * NB_FREQ_RES   # 1.92 MHz

INT16_MAX      = 32767


# ── Noise floor estimation ────────────────────────────────────────────────
def noise_floor_percentile(power: np.ndarray, p: float = 0.30) -> float:
    """30th percentile of strided samples (mirrors C strided-sample approach)."""
    stride   = max(1, len(power) // 1024)
    sampled  = power[::stride]
    return float(np.percentile(sampled, p * 100))


def noise_to_threshold(perc30: float, pfa: float = 1e-4) -> float:
    """Analytical threshold for Exp-distributed power, given 30th-percentile noise."""
    lam = perc30 / (-np.log(0.70))
    return lam * (-np.log(pfa))


# ── Clustering ────────────────────────────────────────────────────────────
def cluster_detections(det_bins: np.ndarray, gap: int):
    """Returns list of (start, end) tuples."""
    if len(det_bins) == 0:
        return []
    clusters, start, end = [], int(det_bins[0]), int(det_bins[0])
    for k in det_bins[1:]:
        k = int(k)
        if k - end <= gap + 1:
            end = k
        else:
            clusters.append((start, end))
            start = end = k
    clusters.append((start, end))
    return clusters


# ── Spectrum generation ───────────────────────────────────────────────────
def add_signal(power: np.ndarray, freq_res: float, n_total: int,
               rf_center: float, sig_cf: float, bw: float, pwr: float):
    cf_bin  = int((sig_cf - rf_center) / freq_res + n_total / 2)
    half_bw = max(1, int(bw / freq_res / 2))
    lo = max(0, cf_bin - half_bw)
    hi = min(n_total - 1, cf_bin + half_bw)
    power[lo:hi + 1] += pwr


def make_wb_spectrum(rf_center: float, signals: list,
                      noise_floor: float = 1e-4) -> np.ndarray:
    rng   = np.random.default_rng(42)
    power = rng.exponential(noise_floor, WB_TOTAL_BINS).astype(np.float32)
    for s in signals:
        add_signal(power, WB_FREQ_RES, WB_TOTAL_BINS,
                   rf_center, s['cf_hz'], s['bw_hz'], s['power_lin'])
    mag   = np.sqrt(power)
    phase = rng.uniform(0, 2 * np.pi, WB_TOTAL_BINS).astype(np.float32)
    return (mag * np.exp(1j * phase)).astype(np.complex64)


def make_nb_spectrum(rf_tune: float, sig_cf: float, bw: float, pwr: float,
                      noise_floor: float = 1e-5) -> np.ndarray:
    rng   = np.random.default_rng(7)
    power = rng.exponential(noise_floor, NB_N_POINTS).astype(np.float32)
    add_signal(power, NB_FREQ_RES, NB_N_POINTS, rf_tune, sig_cf, bw, pwr)
    mag   = np.sqrt(power)
    phase = rng.uniform(0, 2 * np.pi, NB_N_POINTS).astype(np.float32)
    return (mag * np.exp(1j * phase)).astype(np.complex64)


# ── Detectors ────────────────────────────────────────────────────────────
def wb_detect(spectrum: np.ndarray, rf_center: float,
               pfa: float = 1e-4, gap_bins: int = 8):
    power     = (spectrum.real ** 2 + spectrum.imag ** 2).astype(np.float32)
    threshold = noise_to_threshold(noise_floor_percentile(power), pfa)

    det_bins  = np.where(power > threshold)[0]
    clusters  = cluster_detections(det_bins, gap=gap_bins)

    results = []
    for (s, e) in clusters:
        if e - s + 1 < 2:
            continue
        seg     = power[s:e + 1]
        cf_bin  = s + np.average(np.arange(len(seg)), weights=seg)
        cf_hz   = rf_center + (cf_bin - WB_TOTAL_BINS / 2) * WB_FREQ_RES
        bw_hz   = (e - s + 1) * WB_FREQ_RES
        pk_dbfs = 10 * np.log10(seg.max() / 2.0 + 1e-30)
        results.append({'cf_hz': cf_hz, 'bw_hz': bw_hz, 'power_dbfs': pk_dbfs})
    return results


def nb_detect(spectrum: np.ndarray, rf_tune: float,
               pfa: float = 1e-4, gap_bins: int = 8):
    power     = (spectrum.real ** 2 + spectrum.imag ** 2).astype(np.float32)
    threshold = noise_to_threshold(noise_floor_percentile(power), pfa)

    det_bins  = np.where(power > threshold)[0]
    clusters  = cluster_detections(det_bins, gap=gap_bins)
    if not clusters:
        return None

    best = max(clusters, key=lambda c: power[c[0]:c[1] + 1].max())
    s, e = best
    seg     = power[s:e + 1]
    cf_bin  = s + np.average(np.arange(len(seg)), weights=seg)
    cf_hz   = rf_tune + (cf_bin - NB_N_POINTS / 2) * NB_FREQ_RES
    bw_hz   = (e - s + 1) * NB_FREQ_RES
    pk_dbfs = 10 * np.log10(seg.max() / 2.0 + 1e-30)
    return {'cf_hz': cf_hz, 'bw_hz': bw_hz, 'power_dbfs': pk_dbfs}


# ── Binary packers ────────────────────────────────────────────────────────
def pack_wb(spectrum: np.ndarray) -> bytes:
    """Pack complex[32768] → 4096 × [I_ch0..7(int16), Q_ch0..7(int16)]"""
    max_mag = np.abs(spectrum).max()
    if max_mag > 0:
        s = spectrum / (max_mag * 1.01)
    else:
        s = spectrum
    re_i16 = np.clip(s.real * INT16_MAX, -INT16_MAX, INT16_MAX).astype(np.int16)
    im_i16 = np.clip(s.imag * INT16_MAX, -INT16_MAX, INT16_MAX).astype(np.int16)

    buf = bytearray(WB_BINS_PER_CH * 32)
    for lb in range(WB_BINS_PER_CH):
        off = lb * 32
        for ch in range(WB_N_CH):
            g = ch * WB_BINS_PER_CH + lb
            struct.pack_into('<h', buf, off + ch * 2,       re_i16[g])
            struct.pack_into('<h', buf, off + 16 + ch * 2, im_i16[g])
    return bytes(buf)


def pack_nb(spectrum: np.ndarray) -> bytes:
    """Pack complex[2048] → interleaved [I0,Q0,I1,Q1,...] int16 Q15"""
    max_mag = np.abs(spectrum).max()
    s = spectrum / (max_mag * 1.01) if max_mag > 0 else spectrum
    re_i16 = np.clip(s.real * INT16_MAX, -INT16_MAX, INT16_MAX).astype(np.int16)
    im_i16 = np.clip(s.imag * INT16_MAX, -INT16_MAX, INT16_MAX).astype(np.int16)
    interleaved = np.empty(NB_N_POINTS * 2, dtype=np.int16)
    interleaved[0::2] = re_i16
    interleaved[1::2] = im_i16
    return interleaved.tobytes()


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rf_center_mhz', type=float, default=3500.0)
    p.add_argument('--out_dir',       default='detection/testdata')
    args = p.parse_args()

    rf_center = args.rf_center_mhz * 1e6
    os.makedirs(args.out_dir, exist_ok=True)

    signals = [
        {'cf_hz': rf_center + 50e6,   'bw_hz': 400e3,  'power_lin': 0.30},
        {'cf_hz': rf_center - 80e6,   'bw_hz': 1.5e6,  'power_lin': 0.20},
        {'cf_hz': rf_center + 200e6,  'bw_hz': 200e3,  'power_lin': 0.50},
    ]

    print(f"RF center : {rf_center / 1e6:.1f} MHz")
    print(f"WB span   : {WB_TOTAL_BW / 1e6:.1f} MHz  res={WB_FREQ_RES / 1e3:.3f} kHz")
    print(f"NB span   : {NB_BW / 1e6:.3f} MHz  res={NB_FREQ_RES:.1f} Hz\n")

    print("Planted signals:")
    for s in signals:
        offset = (s['cf_hz'] - rf_center) / 1e6
        print(f"  CF={s['cf_hz'] / 1e6:.3f} MHz  ({offset:+.1f} MHz)  "
              f"BW={s['bw_hz'] / 1e3:.1f} kHz  Pwr_lin={s['power_lin']}")

    # ── Wideband ───────────────────────────────────────────────────────
    wb_spec  = make_wb_spectrum(rf_center, signals)
    wb_bytes = pack_wb(wb_spec)
    wb_path  = os.path.join(args.out_dir, 'wb_test.bin')
    with open(wb_path, 'wb') as f:
        f.write(wb_bytes)
    print(f"\nSaved: {wb_path}  ({len(wb_bytes) // 1024} KB)")

    print("\n── Python WB Detection ──")
    wb_results = wb_detect(wb_spec, rf_center)
    if not wb_results:
        print("  No signals found.")
    for r in wb_results:
        off = (r['cf_hz'] - rf_center) / 1e6
        print(f"  CF={r['cf_hz'] / 1e6:.3f} MHz  ({off:+.3f} MHz)  "
              f"BW={r['bw_hz'] / 1e3:.1f} kHz  Pwr={r['power_dbfs']:.1f} dBFS")

    # ── Narrowband (strongest WB signal) ──────────────────────────────
    nb_path = None
    if wb_results:
        strongest = max(wb_results, key=lambda r: r['power_dbfs'])
        nb_tune   = strongest['cf_hz']
        nearest   = min(signals, key=lambda s: abs(s['cf_hz'] - nb_tune))

        nb_spec  = make_nb_spectrum(nb_tune, nearest['cf_hz'],
                                     nearest['bw_hz'], nearest['power_lin'])
        nb_bytes = pack_nb(nb_spec)
        nb_path  = os.path.join(args.out_dir, 'nb_test.bin')
        with open(nb_path, 'wb') as f:
            f.write(nb_bytes)
        print(f"\nSaved: {nb_path}  (8 KB, NB tune={nb_tune / 1e6:.3f} MHz)")

        print("\n── Python NB Detection ──")
        nb_result = nb_detect(nb_spec, nb_tune)
        if nb_result:
            off = (nb_result['cf_hz'] - nb_tune) / 1e3
            print(f"  CF={nb_result['cf_hz'] / 1e6:.6f} MHz  ({off:+.2f} kHz offset)  "
                  f"BW={nb_result['bw_hz'] / 1e3:.3f} kHz  Pwr={nb_result['power_dbfs']:.1f} dBFS")
        else:
            print("  No signal detected.")

    # ── Save scenario JSON ─────────────────────────────────────────────
    sc_path = os.path.join(args.out_dir, 'scenario.json')
    with open(sc_path, 'w') as f:
        json.dump({'rf_center_hz': rf_center, 'label': 'sim_test'}, f, indent=2)
    print(f"\nSaved: {sc_path}")

    if nb_path:
        print(f"\nC test command:")
        print(f"  ./detection/build/detect {sc_path} {wb_path} {nb_path}")


if __name__ == '__main__':
    main()
