# training/simulate.py
import numpy as np

MODULATIONS = [
    'AM', 'FM', 'PM', 'CW',
    '2PSK', '4PSK', '8PSK',
    'DBPSK', 'DQPSK', 'OQPSK',
    '2FSK', '4FSK', '8FSK',
    '8QAM', '16QAM',
    '16APSK', '32APSK', '64APSK', '128APSK', '256APSK',
    'OOK'
]
MOD_TO_IDX = {m: i for i, m in enumerate(MODULATIONS)}


def _apsk_constellation(rings):
    """rings: list of (n_points, radius, phase_offset_rad)"""
    pts = []
    for n, r, offset in rings:
        pts += [r * np.exp(1j * (2 * np.pi * k / n + offset)) for k in range(n)]
    return np.array(pts)


def _normalize(c):
    return c / np.sqrt(np.mean(np.abs(c) ** 2))


_CONSTELLATIONS = {
    '2PSK':    _normalize(np.array([1+0j, -1+0j])),
    '4PSK':    _normalize(np.exp(1j * np.pi / 4 * np.array([1, 3, 5, 7]))),
    '8PSK':    _normalize(np.exp(1j * 2 * np.pi * np.arange(8) / 8)),
    '8QAM':    _normalize(_apsk_constellation([(4, 1.0, np.pi/4), (4, 2.0, 0)])),
    '16QAM':   _normalize(np.array([complex(i, q)
                                    for i in [-3, -1, 1, 3]
                                    for q in [-3, -1, 1, 3]])),
    '16APSK':  _normalize(_apsk_constellation([(4, 1.0, np.pi/4), (12, 2.53, 0)])),
    '32APSK':  _normalize(_apsk_constellation([(4, 1.0, np.pi/4),
                                               (12, 2.84, 0), (16, 5.27, np.pi/16)])),
    '64APSK':  _normalize(_apsk_constellation([(4, 1.0, np.pi/4), (12, 2.4, 0),
                                               (20, 4.3, 0),  (28, 7.0, 0)])),
    '128APSK': _normalize(_apsk_constellation([(4, 1.0, np.pi/4), (12, 2.4, 0),
                                               (24, 4.3, 0),  (40, 6.5, 0),
                                               (48, 9.0, 0)])),
    '256APSK': _normalize(_apsk_constellation([(4,  1.0,  np.pi/4),
                                               (12, 2.4,  0),
                                               (24, 4.3,  0),
                                               (48, 6.5,  0),
                                               (80, 9.0,  0),
                                               (88, 12.0, 0)])),
    'OOK':     np.array([0+0j, 1+0j]),
}


def _sym_to_samples(const, n_samples, sps):
    n_sym = int(np.ceil(n_samples / sps)) + 1
    idx = np.random.randint(0, len(const), n_sym)
    return np.repeat(const[idx], sps)[:n_samples]


def generate_signal(mod, n_samples=1024, sps=8):
    """Generate baseband IQ for one modulation. Returns complex array (n_samples,)."""
    t = np.arange(n_samples)

    if mod == 'CW':
        return np.ones(n_samples, dtype=complex)

    if mod == 'AM':
        msg = np.sin(2 * np.pi * np.random.uniform(0.01, 0.1) * t)
        m   = np.random.uniform(0.5, 1.0)
        return (1.0 + m * msg).astype(complex)

    if mod == 'FM':
        msg   = np.sin(2 * np.pi * np.random.uniform(0.01, 0.05) * t)
        phase = 2 * np.pi * np.random.uniform(0.1, 0.3) * np.cumsum(msg) / n_samples
        return np.exp(1j * phase)

    if mod == 'PM':
        msg = np.sin(2 * np.pi * np.random.uniform(0.01, 0.05) * t)
        return np.exp(1j * np.random.uniform(0.5, 2.0) * msg)

    if mod == 'DBPSK':
        n_sym = int(np.ceil(n_samples / sps)) + 1
        bits  = np.random.randint(0, 2, n_sym)
        phases = np.cumsum(bits * np.pi)
        return np.repeat(np.exp(1j * phases), sps)[:n_samples]

    if mod == 'DQPSK':
        n_sym = int(np.ceil(n_samples / sps)) + 1
        diffs = np.random.randint(0, 4, n_sym)
        phases = np.cumsum(diffs * (np.pi / 2))
        return np.repeat(np.exp(1j * phases), sps)[:n_samples]

    if mod == 'OQPSK':
        # Q stream delayed by sps/2 samples relative to I stream
        half  = sps // 2
        n_sym = int(np.ceil((n_samples + half) / sps)) + 1
        const = _CONSTELLATIONS['4PSK']
        idx   = np.random.randint(0, 4, n_sym)
        syms  = const[idx]
        i_stream = np.repeat(np.real(syms), sps)
        q_stream = np.repeat(np.imag(syms), sps)
        i_out = i_stream[:n_samples]
        q_out = np.concatenate([np.zeros(half), q_stream])[:n_samples]
        return (i_out + 1j * q_out).astype(complex)

    if mod in ('2FSK', '4FSK', '8FSK'):
        order = int(mod[0])
        freqs = np.linspace(-0.15, 0.15, order)
        n_sym = int(np.ceil(n_samples / sps)) + 1
        sym_idx = np.random.randint(0, order, n_sym)
        out, phase, t_idx = np.zeros(n_samples, dtype=complex), 0.0, 0
        for s in sym_idx:
            for _ in range(sps):
                if t_idx >= n_samples:
                    break
                out[t_idx] = np.exp(1j * phase)
                phase += 2 * np.pi * freqs[s]
                t_idx += 1
            if t_idx >= n_samples:
                break
        return out

    return _sym_to_samples(_CONSTELLATIONS[mod], n_samples, sps)


def add_awgn(signal, snr_db):
    """Add AWGN at target SNR (dB). SNR = signal_power / noise_power."""
    sig_power  = np.mean(np.abs(signal) ** 2)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_std  = np.sqrt(sig_power / snr_linear / 2)
    noise      = noise_std * (np.random.randn(len(signal)) +
                              1j * np.random.randn(len(signal)))
    return signal + noise
