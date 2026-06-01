# training/dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset
from simulate import generate_signal, add_awgn, MODULATIONS


def _preprocess(i_arr, q_arr):
    """DC removal -> power normalization -> instantaneous features.
    Returns float32 ndarray of shape (3, 1024).
    Must match the logic in inference/preprocess.c exactly."""
    i = i_arr - np.mean(i_arr)
    q = q_arr - np.mean(q_arr)

    power = np.mean(i ** 2 + q ** 2)
    if power > 1e-10:
        s = 1.0 / np.sqrt(power)
        i, q = i * s, q * s

    amp  = np.sqrt(i ** 2 + q ** 2).astype(np.float32)
    phi  = np.arctan2(q, i).astype(np.float32)
    dphi = np.diff(np.unwrap(phi), prepend=phi[0]).astype(np.float32)
    dphi[0] = 0.0

    return np.stack([amp, phi, dphi])   # (3, 1024)


def _augment(i_arr, q_arr):
    """Apply random RF impairments in-place. Returns augmented (i, q)."""
    n = len(i_arr)
    t = np.arange(n, dtype=np.float32)

    # Frequency offset +-5% of normalized fs=1
    f_off  = np.random.uniform(-0.05, 0.05)
    iq     = (i_arr + 1j * q_arr) * np.exp(1j * 2 * np.pi * f_off * t)
    i_arr, q_arr = np.real(iq).astype(np.float32), np.imag(iq).astype(np.float32)

    # Phase offset uniform [0, 2pi]
    phi_off = np.random.uniform(0, 2 * np.pi)
    iq      = (i_arr + 1j * q_arr) * np.exp(1j * phi_off)
    i_arr, q_arr = np.real(iq).astype(np.float32), np.imag(iq).astype(np.float32)

    # IQ amplitude imbalance +-1 dB
    amp_imb = 10 ** (np.random.uniform(-1.0, 1.0) / 20.0)
    i_arr  *= amp_imb

    # IQ phase imbalance +-5 degrees
    phi_imb  = np.random.uniform(-5.0, 5.0) * np.pi / 180.0
    i_new    = i_arr * np.cos(phi_imb) - q_arr * np.sin(phi_imb)
    q_new    = i_arr * np.sin(phi_imb) + q_arr * np.cos(phi_imb)

    return i_new, q_new


class AMCDataset(Dataset):
    def __init__(self, n_per_class_per_snr=500,
                 snr_range=(-10, 20), snr_step=2,
                 n_samples=1024, sps=8, augment=True):
        snrs = range(snr_range[0], snr_range[1] + 1, snr_step)
        self.augment = augment
        self._items = []
        for label, mod in enumerate(MODULATIONS):
            for snr in snrs:
                for _ in range(n_per_class_per_snr):
                    iq = add_awgn(generate_signal(mod, n_samples, sps), snr)
                    self._items.append((
                        np.real(iq).astype(np.float32),
                        np.imag(iq).astype(np.float32),
                        label,
                    ))

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        i_arr, q_arr, label = self._items[idx]
        if self.augment:
            i_arr, q_arr = _augment(i_arr.copy(), q_arr.copy())
        return torch.from_numpy(_preprocess(i_arr, q_arr)), label
