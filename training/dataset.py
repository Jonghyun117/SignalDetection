# training/dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset
from simulate import generate_signal, add_awgn, MODULATIONS, CLASSES, MOD_TO_CLASS_IDX, FS

# SPS range derived from FS=512 kHz and symbol rate [75, 64000] baud
_LOG_SPS_MIN = np.log(8)      # 64000 baud
_LOG_SPS_MAX = np.log(6827)   # 75 baud


def _preprocess(i_arr, q_arr):
    """DC removal → power normalization → instantaneous features.

    Channel 0 (amp): raw amplitude BEFORE DC removal.
        Preserves OOK on/off envelope and AM sinusoidal envelope,
        which DC removal would otherwise destroy.
    Channel 1 (phi): instantaneous phase on DC-removed, normalized signal.
    Channel 2 (dphi): instantaneous frequency (diff of unwrapped phi).

    Returns float32 ndarray of shape (3, 1024).
    Must match the logic in inference/preprocess.c exactly.
    """
    # Raw amplitude before DC removal
    raw_amp = np.sqrt(i_arr ** 2 + q_arr ** 2)
    amp_pwr = np.mean(raw_amp ** 2)
    amp     = (raw_amp / np.sqrt(amp_pwr) if amp_pwr > 1e-10 else raw_amp).astype(np.float32)

    # DC removal
    i = i_arr - np.mean(i_arr)
    q = q_arr - np.mean(q_arr)

    # Power normalization
    power = np.mean(i ** 2 + q ** 2)
    if power > 1e-10:
        s = 1.0 / np.sqrt(power)
        i, q = i * s, q * s

    phi  = np.arctan2(q, i).astype(np.float32)
    dphi = np.diff(np.unwrap(phi), prepend=phi[0]).astype(np.float32)
    dphi[0] = 0.0

    return np.stack([amp, phi, dphi])   # (3, 1024)


def _augment(i_arr, q_arr):
    """Random RF impairments: freq offset, phase offset, IQ imbalance."""
    n   = len(i_arr)
    t   = np.arange(n, dtype=np.float32)

    # Frequency offset ±5% of normalized fs
    f_off = np.random.uniform(-0.05, 0.05)
    iq    = (i_arr + 1j * q_arr) * np.exp(1j * 2 * np.pi * f_off * t)
    i_arr, q_arr = np.real(iq).astype(np.float32), np.imag(iq).astype(np.float32)

    # Phase offset uniform [0, 2π]
    phi_off = np.random.uniform(0, 2 * np.pi)
    iq      = (i_arr + 1j * q_arr) * np.exp(1j * phi_off)
    i_arr, q_arr = np.real(iq).astype(np.float32), np.imag(iq).astype(np.float32)

    # IQ amplitude imbalance ±1 dB
    i_arr *= 10 ** (np.random.uniform(-1.0, 1.0) / 20.0)

    # IQ phase imbalance ±5 degrees
    phi_imb = np.random.uniform(-5.0, 5.0) * np.pi / 180.0
    i_new   = i_arr * np.cos(phi_imb) - q_arr * np.sin(phi_imb)
    q_new   = i_arr * np.sin(phi_imb) + q_arr * np.cos(phi_imb)

    return i_new, q_new


class AMCDataset(Dataset):
    def __init__(self, n_per_class_per_snr=500,
                 snr_range=(-10, 20), snr_step=2,
                 n_samples=1024, augment=True):
        snrs = list(range(snr_range[0], snr_range[1] + 1, snr_step))
        self.augment   = augment
        self.n_samples = n_samples
        # Lazy: store (mod, snr, class_idx) — signal generated on-the-fly.
        # 2PSK and DBPSK share the BPSK/DBPSK label; 4PSK and DQPSK share QPSK/DQPSK.
        self._items = [
            (mod, snr, MOD_TO_CLASS_IDX[mod])
            for mod in MODULATIONS
            for snr in snrs
            for _ in range(n_per_class_per_snr)
        ]

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        mod, snr, label = self._items[idx]

        # Log-uniform symbol rate in [75, 64000] baud → SPS in [8, 6827]
        sps      = float(np.exp(np.random.uniform(_LOG_SPS_MIN, _LOG_SPS_MAX)))
        roll_off = float(np.random.uniform(0.5, 1.0))

        iq    = add_awgn(generate_signal(mod, self.n_samples, sps, roll_off), snr)
        i_arr = np.real(iq).astype(np.float32)
        q_arr = np.imag(iq).astype(np.float32)

        if self.augment:
            i_arr, q_arr = _augment(i_arr, q_arr)

        return torch.from_numpy(_preprocess(i_arr, q_arr)), label
