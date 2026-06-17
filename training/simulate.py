# training/simulate.py
import numpy as np
from scipy.signal import fftconvolve


MODULATIONS = [
    'AM', 'FM', 'PM', 'CW',
    '2PSK', '4PSK', '8PSK',
    'DBPSK', 'DQPSK', 'OQPSK',
    '2FSK', '4FSK', '8FSK',
    '8QAM', '16QAM',
    '16APSK', '32APSK', '64APSK', '128APSK', '256APSK',
    'OOK',
]

# Output classes: 2PSK+DBPSK share one label, 4PSK+DQPSK share one label.
CLASSES = [
    'AM', 'FM', 'PM', 'CW',
    'BPSK/DBPSK',          # 2PSK and DBPSK are indistinguishable
    'QPSK/DQPSK',          # 4PSK and DQPSK are indistinguishable
    '8PSK', 'OQPSK',
    '2FSK', '4FSK', '8FSK',
    '8QAM', '16QAM',
    '16APSK', '32APSK', '64APSK', '128APSK', '256APSK',
    'OOK',
]

_CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
_MOD_TO_CLASS = {m: m for m in MODULATIONS}
_MOD_TO_CLASS['2PSK']  = 'BPSK/DBPSK'
_MOD_TO_CLASS['DBPSK'] = 'BPSK/DBPSK'
_MOD_TO_CLASS['4PSK']  = 'QPSK/DQPSK'
_MOD_TO_CLASS['DQPSK'] = 'QPSK/DQPSK'

MOD_TO_CLASS_IDX = {m: _CLASS_TO_IDX[_MOD_TO_CLASS[m]] for m in MODULATIONS}


# ── Constellation tables ──────────────────────────────────────────────────────

def _apsk_constellation(rings):
    pts = []
    for n, r, offset in rings:
        pts += [r * np.exp(1j * (2 * np.pi * k / n + offset)) for k in range(n)]
    return np.array(pts)


def _normalize(c):
    return c / np.sqrt(np.mean(np.abs(c) ** 2))


_CONSTELLATIONS = {
    '2PSK':   _normalize(np.array([1+0j, -1+0j])),
    '4PSK':   _normalize(np.exp(1j * np.pi / 4 * np.array([1, 3, 5, 7]))),
    '8PSK':   _normalize(np.exp(1j * 2 * np.pi * np.arange(8) / 8)),
    '8QAM':   _normalize(_apsk_constellation([(4, 1.0, np.pi/4), (4, 2.0, 0)])),
    '16QAM':  _normalize(np.array([complex(i, q)
                                   for i in [-3, -1, 1, 3]
                                   for q in [-3, -1, 1, 3]])),
    '16APSK': _normalize(_apsk_constellation([(4, 1.0, np.pi/4), (12, 2.53, 0)])),
    '32APSK': _normalize(_apsk_constellation([(4, 1.0, np.pi/4),
                                              (12, 2.84, 0), (16, 5.27, np.pi/16)])),
    '64APSK': _normalize(_apsk_constellation([(4, 1.0, np.pi/4), (12, 2.4, 0),
                                              (20, 4.3, 0),  (28, 7.0, 0)])),
    '128APSK':_normalize(_apsk_constellation([(4, 1.0, np.pi/4), (12, 2.4, 0),
                                              (24, 4.3, 0),  (40, 6.5, 0),
                                              (48, 9.0, 0)])),
    '256APSK':_normalize(_apsk_constellation([(4,  1.0,  np.pi/4),
                                              (12, 2.4,  0),
                                              (24, 4.3,  0),
                                              (48, 6.5,  0),
                                              (80, 9.0,  0),
                                              (88, 12.0, 0)])),
    'OOK':    np.array([0+0j, 1+0j]),
}

# ── RRC pulse shaping ─────────────────────────────────────────────────────────

def _rrc_filter(sps, roll_off):
    """Root Raised Cosine filter coefficients. sps: int, roll_off in [0,1]."""
    β = float(roll_off)
    span = max(2, min(6, 256 // max(1, sps)))
    n_taps = 2 * sps * span + 1
    n = (n_taps - 1) // 2
    t = np.arange(-n, n + 1, dtype=float) / sps
    h = np.zeros(len(t))
    for i, τ in enumerate(t):
        d = 1.0 - (4.0 * β * τ) ** 2
        if abs(τ) < 1e-8:
            h[i] = 1.0 + β * (4.0 / np.pi - 1.0)
        elif abs(d) < 1e-6:
            h[i] = (β / np.sqrt(2.0)) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * β)) +
                (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * β))
            )
        else:
            h[i] = (np.sin(np.pi * τ * (1.0 - β)) +
                    4.0 * β * τ * np.cos(np.pi * τ * (1.0 + β))) / (np.pi * τ * d)
    h /= np.sqrt(np.sum(h ** 2))
    return h.astype(np.float64)


def _pulse_shape(syms, sps_int, roll_off, n_samples):
    """Upsample symbols, apply RRC, return first n_samples."""
    up = np.zeros(len(syms) * sps_int, dtype=complex)
    up[::sps_int] = syms
    h = _rrc_filter(sps_int, roll_off)
    shaped = (fftconvolve(up.real, h, mode='same') +
              1j * fftconvolve(up.imag, h, mode='same'))
    if len(shaped) >= n_samples:
        start = (len(shaped) - n_samples) // 2
        return shaped[start: start + n_samples]
    return np.resize(shaped, n_samples)


def _sym_to_samples(const, n_samples, sps_int, roll_off):
    n_sym = max(4, int(np.ceil(n_samples / sps_int)) + 2)
    syms = const[np.random.randint(0, len(const), n_sym)]
    return _pulse_shape(syms, sps_int, roll_off, n_samples)


# ── Signal generation ─────────────────────────────────────────────────────────

def generate_signal(mod, n_samples=1024, sps=4, roll_off=0.35):
    """Generate baseband IQ for one modulation.

    sps      : samples per symbol (float, [8, 6827] for 64k–75 baud at 512 kHz)
    roll_off : RRC roll-off factor [0.5, 1.0]
    Returns  : complex ndarray (n_samples,)
    """
    sps_int = max(1, int(round(sps)))
    t = np.arange(n_samples, dtype=float)

    # ── Analog modulations ────────────────────────────────────────────────────
    if mod == 'CW':
        # Single carrier at random offset so DC-removal doesn't kill it.
        # dphi = constant → unique signature vs FSK (dphi switches values)
        f_c = np.random.uniform(0.05, 0.20)
        return np.exp(1j * 2 * np.pi * f_c * t)

    if mod == 'AM':
        # AM with carrier: raw amplitude shows sinusoidal envelope,
        # clearly distinguishable from CW (flat amp) and PM (flat amp).
        f_c   = np.random.uniform(0.05, 0.15)
        f_msg = np.random.uniform(0.005, 0.03)
        m     = np.random.uniform(0.5, 1.0)
        msg   = np.sin(2 * np.pi * f_msg * t)
        return (1.0 + m * msg) * np.exp(1j * 2 * np.pi * f_c * t)

    if mod == 'FM':
        f_msg = np.random.uniform(0.005, 0.04)
        f_dev = np.random.uniform(0.1, 0.3)
        msg   = np.sin(2 * np.pi * f_msg * t)
        phase = 2 * np.pi * f_dev * np.cumsum(msg) / n_samples
        return np.exp(1j * phase)

    if mod == 'PM':
        f_msg = np.random.uniform(0.005, 0.04)
        msg   = np.sin(2 * np.pi * f_msg * t)
        return np.exp(1j * np.random.uniform(0.5, 2.0) * msg)

    # ── Digital modulations with RRC pulse shaping ────────────────────────────
    if mod == '2PSK':
        return _sym_to_samples(_CONSTELLATIONS['2PSK'], n_samples, sps_int, roll_off)

    if mod == 'DBPSK':
        n_sym  = max(4, int(np.ceil(n_samples / sps_int)) + 2)
        bits   = np.random.randint(0, 2, n_sym)
        phases = np.cumsum(bits * np.pi)
        syms   = np.exp(1j * phases)
        return _pulse_shape(syms, sps_int, roll_off, n_samples)

    if mod == '4PSK':
        return _sym_to_samples(_CONSTELLATIONS['4PSK'], n_samples, sps_int, roll_off)

    if mod == 'DQPSK':
        n_sym  = max(4, int(np.ceil(n_samples / sps_int)) + 2)
        diffs  = np.random.randint(0, 4, n_sym)
        phases = np.cumsum(diffs * (np.pi / 2))
        syms   = np.exp(1j * phases)
        return _pulse_shape(syms, sps_int, roll_off, n_samples)

    if mod == 'OQPSK':
        half  = sps_int // 2
        n_sym = max(4, int(np.ceil((n_samples + half) / sps_int)) + 2)
        syms  = _CONSTELLATIONS['4PSK'][np.random.randint(0, 4, n_sym)]
        h     = _rrc_filter(sps_int, roll_off)
        i_up  = np.zeros(n_sym * sps_int)
        q_up  = np.zeros(n_sym * sps_int)
        i_up[::sps_int] = syms.real
        q_up[::sps_int] = syms.imag
        i_f   = fftconvolve(i_up, h, mode='same')
        q_f   = fftconvolve(q_up, h, mode='same')
        q_del = np.concatenate([np.zeros(half), q_f])
        i_out = i_f[:n_samples]  if len(i_f)   >= n_samples else np.resize(i_f,   n_samples)
        q_out = q_del[:n_samples] if len(q_del) >= n_samples else np.resize(q_del, n_samples)
        return (i_out + 1j * q_out).astype(complex)

    if mod in ('2FSK', '4FSK', '8FSK'):
        order    = int(mod[0])
        freqs    = np.linspace(-0.15, 0.15, order)
        n_sym    = max(4, int(np.ceil(n_samples / sps_int)) + 2)
        sym_idx  = np.random.randint(0, order, n_sym)
        f_arr    = np.repeat(freqs[sym_idx], sps_int)
        # Continuous-phase FSK: accumulate phase
        phase    = np.cumsum(2 * np.pi * f_arr[:n_samples])
        return np.exp(1j * phase)

    # PSK, QAM, APSK, OOK — all use RRC
    return _sym_to_samples(_CONSTELLATIONS[mod], n_samples, sps_int, roll_off)


def add_awgn(signal, snr_db):
    """Add AWGN at target SNR (dB). SNR = signal_power / noise_power."""
    sig_power  = np.mean(np.abs(signal) ** 2)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_std  = np.sqrt(sig_power / snr_linear / 2)
    noise      = noise_std * (np.random.randn(len(signal)) +
                              1j * np.random.randn(len(signal)))
    return signal + noise
