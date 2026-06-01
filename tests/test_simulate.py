# tests/test_simulate.py
import numpy as np
import sys
sys.path.insert(0, 'training')
from simulate import generate_signal, add_awgn, MODULATIONS

def test_output_shape():
    for mod in MODULATIONS:
        iq = generate_signal(mod, n_samples=1024)
        assert iq.shape == (1024,), f"{mod}: expected (1024,), got {iq.shape}"

def test_bpsk_real_valued():
    iq = generate_signal('2PSK', n_samples=1024)
    assert np.max(np.abs(np.imag(iq))) < 0.01, "2PSK should be real-valued"

def test_psk_unit_circle():
    for mod in ['2PSK', '4PSK', '8PSK']:
        iq = generate_signal(mod, n_samples=1024)
        assert np.allclose(np.abs(iq), 1.0, atol=0.05), f"{mod} not on unit circle"

def test_awgn_snr():
    np.random.seed(0)
    signal = np.ones(4096, dtype=complex)
    noisy = add_awgn(signal, snr_db=0.0)
    noise_power = np.mean(np.abs(noisy - signal) ** 2)
    snr = 10 * np.log10(1.0 / noise_power)
    assert abs(snr) < 2.0, f"SNR mismatch: {snr:.2f} dB"

def test_ook_binary_amplitude():
    iq = generate_signal('OOK', n_samples=1024)
    amps = np.round(np.abs(iq), 1)
    assert set(np.unique(amps)).issubset({0.0, 1.0}), \
        f"OOK should have only 0/1 amplitude"
