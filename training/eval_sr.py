#!/usr/bin/env python3
"""
training/eval_sr.py
심볼률 추정 정확도 평가 — sr_estimate.c의 Method A/B Python 재현

Usage (프로젝트 루트에서):
    python training/eval_sr.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import font_manager
from simulate import generate_signal, add_awgn, MODULATIONS, CLASSES

# 한글 폰트 설정 (Windows: Malgun Gothic)
for _fn in ['Malgun Gothic', 'NanumGothic', 'AppleGothic', 'DejaVu Sans']:
    if any(_fn.lower() in f.name.lower() for f in font_manager.fontManager.ttflist):
        matplotlib.rcParams['font.family'] = _fn
        break
matplotlib.rcParams['axes.unicode_minus'] = False

# ─── 상수 (sr_estimate.c와 동일) ────────────────────────────────────────────
N            = 2048
QUALITY_THR  = 3.0   # peak/mean 비율 최소치
SPS          = 4     # 기준 SPS (1/SPS = 참 심볼률 정규화값)
TRUE_RATE    = 1.0 / SPS   # 0.25
SNR_DB       = np.arange(-5, 31, 5)   # -5, 0, 5 … 30 dB
N_TRIALS     = 60

# ─── C 코드 동등 추정 함수 ──────────────────────────────────────────────────

def _prep_and_fft(x: np.ndarray) -> np.ndarray:
    """평균 제거 → Hann → FFT → |F|² 반환 (길이 N//2+1)."""
    x = x.astype(float)
    x -= x.mean()
    w  = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(N) / (N - 1)))
    x *= w
    return np.abs(np.fft.rfft(x, n=N)) ** 2


def _peak_quality(mag2: np.ndarray, end: int = None):
    """bins [1, end] 에서 피크 빈과 품질비 반환. end 기본값 = N//2 - 1."""
    if end is None:
        end = N // 2 - 1
    seg   = mag2[1 : end + 1]
    k_rel = int(np.argmax(seg))
    bk    = k_rel + 1
    bm    = seg[k_rel]
    rest  = np.concatenate([seg[:k_rel], seg[k_rel + 1:]])
    mean_rest = float(rest.mean()) if len(rest) else 0.0
    quality   = bm / mean_rest if mean_rest > 0.0 else 0.0
    return bk, quality


def estimate_power(I, Q):
    """Method A: |r|² 파워 스펙트럼 — PSK / QAM / APSK."""
    mag2 = _prep_and_fft(I**2 + Q**2)
    k, q = _peak_quality(mag2)          # search [1, N/2-1]
    return float(k) / N, q


def estimate_power_oqpsk(I, Q):
    """Method A': OQPSK 전용 — I/Q 채널 분리 비간섭 합.

    |I|²+|Q|²의 f_sym 성분은 위상 π 차이로 상쇄(OQPSK Q-branch 반주기 지연).
    I²(t)와 Q²(t)는 각각 독립적으로 f_sym 스펙트럼 선을 가지므로,
    |FFT(I²)|² + |FFT(Q²)|² (비간섭 합)으로 피크를 찾는다.
    """
    mag2_i = _prep_and_fft(I**2)
    mag2_q = _prep_and_fft(Q**2)
    combined = mag2_i + mag2_q
    k, q = _peak_quality(combined)      # standard search [1, N//2-1]
    return float(k) / N, q


def estimate_fsk(I, Q):
    """Method B: |Δdphi| 위상가속 스펙트럼 — FSK."""
    z    = I + 1j * Q
    dphi = np.angle(z[1:] * np.conj(z[:-1]))          # 순시 주파수 (N-1)
    prev = np.concatenate([[0.0], dphi[:-1]])           # C의 prev_dphi 구조 동일
    dd   = dphi - prev
    dd   = (dd + np.pi) % (2.0 * np.pi) - np.pi       # [-π, π] wrap
    sig  = np.zeros(N)
    sig[:N - 1] = np.abs(dd)
    mag2 = _prep_and_fft(sig)
    k, q = _peak_quality(mag2)
    return float(k) / N, q


def sr_estimate(I, Q, class_idx: int):
    """amc_sr_estimate() Python 등가. (추정값, 품질비) 반환."""
    if class_idx in (0, 1, 2, 3, 18):   # AM FM PM CW OOK
        return 0.0, 0.0
    if class_idx in (8, 9, 10):          # 2FSK 4FSK 8FSK
        return estimate_fsk(I, Q)
    if class_idx == 7:                   # OQPSK: 2×f_sym 피크 → ÷2
        return estimate_power_oqpsk(I, Q)
    return estimate_power(I, Q)          # PSK / QAM / APSK

# ─── 클래스 매핑 ─────────────────────────────────────────────────────────────
_MOD2CLS = {m: m for m in MODULATIONS}
_MOD2CLS.update({'2PSK': 'BPSK/DBPSK', 'DBPSK': 'BPSK/DBPSK',
                 '4PSK': 'QPSK/DQPSK', 'DQPSK': 'QPSK/DQPSK'})
_CLS_IDX   = {c: i for i, c in enumerate(CLASSES)}
_MOD_IDX   = {m: _CLS_IDX[_MOD2CLS[m]] for m in MODULATIONS}

# 클래스당 대표 변조 1개, 추정 불가(AM/FM/PM/CW/OOK) 제외
_seen, EVAL_MODS, EVAL_LABELS, EVAL_CIDX = set(), [], [], []
for m in MODULATIONS:
    cls, idx = _MOD2CLS[m], _MOD_IDX[m]
    if idx not in (0, 1, 2, 3, 18) and cls not in _seen:
        _seen.add(cls)
        EVAL_MODS.append(m)
        EVAL_LABELS.append(cls)
        EVAL_CIDX.append(idx)

M, S = len(EVAL_MODS), len(SNR_DB)

# ─── 평가 루프 ────────────────────────────────────────────────────────────────
mean_err   = np.full((M, S), np.nan)
p10_err    = np.full((M, S), np.nan)   # 90th‑percentile error
detect_rt  = np.zeros((M, S))

t0 = time.time()
print(f"[eval_sr] {M}개 변조 × {S}개 SNR × {N_TRIALS}회 시행")
print(f"  SPS={SPS}  TRUE_RATE={TRUE_RATE:.4f}  QUALITY_THR={QUALITY_THR}")
print()

for mi, (mod, label, cidx) in enumerate(zip(EVAL_MODS, EVAL_LABELS, EVAL_CIDX)):
    method = 'FSK(B)' if cidx in (8, 9, 10) else 'Power(A)'
    print(f"  [{mi+1:2d}/{M}] {label:<14} ({method}) ...", end='', flush=True)
    for si, snr in enumerate(SNR_DB):
        errs, dets = [], 0
        for _ in range(N_TRIALS):
            sig = generate_signal(mod, n_samples=N, sps=SPS, roll_off=0.35)
            sig = add_awgn(sig, snr)
            est, qual = sr_estimate(sig.real, sig.imag, cidx)
            if qual >= QUALITY_THR and est > 0.0:
                dets += 1
                errs.append(abs(est - TRUE_RATE) / TRUE_RATE * 100.0)
        detect_rt[mi, si] = dets / N_TRIALS
        if errs:
            mean_err[mi, si] = float(np.mean(errs))
            p10_err[mi, si]  = float(np.percentile(errs, 90))
    print(f" done  (SNR 30dB: err={mean_err[mi,-1]:.1f}%  det={detect_rt[mi,-1]*100:.0f}%)")

print(f"\n완료: {time.time()-t0:.1f}s")

# ─── 그룹 분리 ───────────────────────────────────────────────────────────────
IDX_A = [i for i, c in enumerate(EVAL_CIDX) if c not in (8, 9, 10)]  # Method A
IDX_B = [i for i, c in enumerate(EVAL_CIDX) if c in (8, 9, 10)]       # Method B

# ─── Figure ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor('#f8f9fa')
fig.suptitle(
    f'심볼률 추정 정확도  (SPS={SPS}, N={N}, 시행={N_TRIALS}회/점)\n'
    f'참 심볼률 = 1/{SPS} = {TRUE_RATE:.3f} fs',
    fontsize=13, fontweight='bold', y=0.97
)

gs = gridspec.GridSpec(2, 3, figure=fig,
                        hspace=0.45, wspace=0.38,
                        left=0.07, right=0.97, top=0.91, bottom=0.07)

ax_A   = fig.add_subplot(gs[0, 0])
ax_B   = fig.add_subplot(gs[0, 1])
ax_dr  = fig.add_subplot(gs[0, 2])
ax_he  = fig.add_subplot(gs[1, 0:2])
ax_hd  = fig.add_subplot(gs[1, 2])

cmap20 = plt.cm.tab20
_FMT = dict(markersize=5, linewidth=1.4)

# ── (a) Method A 오차 선 그래프 ───────────────────────────────────────────────
for j, i in enumerate(IDX_A):
    c = cmap20(j / max(len(IDX_A), 1))
    ax_A.plot(SNR_DB, mean_err[i], marker='o', color=c, label=EVAL_LABELS[i], **_FMT)
    ax_A.fill_between(SNR_DB,
                      np.where(np.isnan(mean_err[i]), 0, mean_err[i]),
                      np.where(np.isnan(p10_err[i]),  0, p10_err[i]),
                      color=c, alpha=0.12)
ax_A.axhline(10, ls='--', color='crimson', lw=1.0, alpha=0.8, label='10% 기준선')
ax_A.set_title('(a) Method A: 파워 스펙트럼\n    PSK / QAM / APSK', fontsize=10)
ax_A.set_xlabel('SNR (dB)')
ax_A.set_ylabel('평균 상대오차 (%)')
ax_A.set_ylim(bottom=0)
ax_A.legend(fontsize=7, ncol=2, loc='upper right', framealpha=0.9)
ax_A.grid(True, alpha=0.35)
ax_A.set_facecolor('#fdfdfd')

# ── (b) Method B 오차 선 그래프 ───────────────────────────────────────────────
fsk_colors = ['#e15759', '#4e79a7', '#59a14f']
for j, i in enumerate(IDX_B):
    c = fsk_colors[j % len(fsk_colors)]
    ax_B.plot(SNR_DB, mean_err[i], marker='s', color=c, label=EVAL_LABELS[i], **_FMT)
    ax_B.fill_between(SNR_DB,
                      np.where(np.isnan(mean_err[i]), 0, mean_err[i]),
                      np.where(np.isnan(p10_err[i]),  0, p10_err[i]),
                      color=c, alpha=0.15)
ax_B.axhline(10, ls='--', color='crimson', lw=1.0, alpha=0.8)
ax_B.set_title('(b) Method B: 위상가속 스펙트럼\n    FSK (CPFSK)', fontsize=10)
ax_B.set_xlabel('SNR (dB)')
ax_B.set_ylabel('평균 상대오차 (%)')
ax_B.set_ylim(bottom=0)
ax_B.legend(fontsize=9, loc='upper right', framealpha=0.9)
ax_B.grid(True, alpha=0.35)
ax_B.set_facecolor('#fdfdfd')

# ── (c) 검출률 선 그래프 ──────────────────────────────────────────────────────
for j in range(M):
    ls = '--' if EVAL_CIDX[j] in (8, 9, 10) else '-'
    c  = cmap20(j / M)
    ax_dr.plot(SNR_DB, detect_rt[j] * 100, ls=ls, marker='.', markersize=4,
               color=c, linewidth=1.3, label=EVAL_LABELS[j])
ax_dr.axhline(90, ls=':', color='gray', lw=0.9, alpha=0.8)
ax_dr.set_title('(c) 검출률 (품질 게이트 통과율)\n    실선=Method A  점선=Method B', fontsize=10)
ax_dr.set_xlabel('SNR (dB)')
ax_dr.set_ylabel('검출률 (%)')
ax_dr.set_ylim(-2, 105)
ax_dr.legend(fontsize=6.5, ncol=2, loc='lower right', framealpha=0.9)
ax_dr.grid(True, alpha=0.35)
ax_dr.set_facecolor('#fdfdfd')

# ── (d) 오차 히트맵 ───────────────────────────────────────────────────────────
err_vis = np.where(np.isnan(mean_err), 100.0, mean_err)   # NaN → 100 (미검출 표시)
im_e = ax_he.imshow(err_vis, aspect='auto', cmap='RdYlGn_r',
                    vmin=0, vmax=30, origin='upper',
                    interpolation='nearest')
ax_he.set_xticks(range(S))
ax_he.set_xticklabels([f'{int(v)}' for v in SNR_DB], fontsize=9)
ax_he.set_yticks(range(M))
ax_he.set_yticklabels(EVAL_LABELS, fontsize=9)
ax_he.set_xlabel('SNR (dB)', fontsize=9)
ax_he.set_title(f'(d) 평균 상대오차 (%)  — "—" = 미검출 (품질미달)', fontsize=10)
cb_e = plt.colorbar(im_e, ax=ax_he, fraction=0.018, pad=0.02)
cb_e.set_label('%', fontsize=8)

# 셀 주석
for i in range(M):
    for j in range(S):
        val = mean_err[i, j]
        txt = '—' if np.isnan(val) else f'{val:.1f}'
        dark = (err_vis[i, j] > 18) or np.isnan(mean_err[i, j])
        ax_he.text(j, i, txt, ha='center', va='center', fontsize=7,
                   color='white' if dark else '#111111', fontweight='bold')

# 그룹 구분선 (Method A / B)
if IDX_B:
    boundary = IDX_B[0] - 0.5
    ax_he.axhline(boundary, color='navy', lw=1.5, ls='--', alpha=0.6)
    ax_he.text(S - 0.3, boundary - 0.3, 'Method A', fontsize=7,
               color='navy', ha='right', va='bottom')
    ax_he.text(S - 0.3, boundary + 0.3, 'Method B', fontsize=7,
               color='navy', ha='right', va='top')

# ── (e) 검출률 히트맵 ─────────────────────────────────────────────────────────
im_d = ax_hd.imshow(detect_rt * 100, aspect='auto', cmap='Blues',
                    vmin=0, vmax=100, origin='upper',
                    interpolation='nearest')
ax_hd.set_xticks(range(S))
ax_hd.set_xticklabels([f'{int(v)}' for v in SNR_DB], fontsize=9)
ax_hd.set_yticks(range(M))
ax_hd.set_yticklabels(EVAL_LABELS, fontsize=9)
ax_hd.set_xlabel('SNR (dB)', fontsize=9)
ax_hd.set_title('(e) 검출률 (%)', fontsize=10)
cb_d = plt.colorbar(im_d, ax=ax_hd, fraction=0.04, pad=0.02)
cb_d.set_label('%', fontsize=8)
for i in range(M):
    for j in range(S):
        val = detect_rt[i, j] * 100
        ax_hd.text(j, i, f'{val:.0f}', ha='center', va='center', fontsize=7,
                   color='white' if val > 55 else '#111111', fontweight='bold')

# ─── 저장 ────────────────────────────────────────────────────────────────────
out_dir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'docs')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'sr_eval.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"저장 완료: {out_path}")
