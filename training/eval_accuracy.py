#!/usr/bin/env python3
"""
training/eval_accuracy.py
SNR별 분류 정확도 평가 및 시각화 (PyTorch 직접 사용, ONNX 불필요)

Usage:
    python training/eval_accuracy.py [--weights model/best.pth] [--n 200]
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

for _fn in ['Malgun Gothic', 'NanumGothic', 'AppleGothic', 'DejaVu Sans']:
    if any(_fn.lower() in f.name.lower() for f in font_manager.fontManager.ttflist):
        matplotlib.rcParams['font.family'] = _fn
        break
matplotlib.rcParams['axes.unicode_minus'] = False

from model import AMCNet
from dataset import _preprocess, _SPS_BASE, _SPS_ERR_MAX
from simulate import generate_signal, add_awgn, MODULATIONS, CLASSES, MOD_TO_CLASS_IDX


# 변조 방식 그룹 (그래프 색 구분용)
GROUPS = {
    'Analog':   ['AM', 'FM', 'PM', 'CW'],
    'PSK':      ['2PSK', 'DBPSK', '4PSK', 'DQPSK', '8PSK', 'OQPSK'],
    'FSK':      ['2FSK', '4FSK', '8FSK'],
    'QAM/APSK': ['8QAM', '16QAM', '16APSK', '32APSK', '64APSK', '128APSK', '256APSK'],
    'OOK':      ['OOK'],
}
MOD_TO_GROUP = {m: g for g, mods in GROUPS.items() for m in mods}
GROUP_COLORS = {'Analog': '#e15759', 'PSK': '#4e79a7', 'FSK': '#59a14f',
                'QAM/APSK': '#f28e2b', 'OOK': '#b07aa1'}


def evaluate(weights_path: str, n_per: int):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AMCNet(num_classes=len(CLASSES))
    model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=False))
    model.eval().to(device)

    snrs = list(range(-10, 21, 2))   # -10 ~ +20 dB, 2 dB 간격

    # results[snr] = [correct, total]
    results     = {snr: [0, 0] for snr in snrs}
    per_mod_acc = {mod: {snr: [0, 0] for snr in snrs} for mod in MODULATIONS}

    total_samples = len(MODULATIONS) * len(snrs) * n_per
    done = 0

    with torch.no_grad():
        for mod in MODULATIONS:
            true_label = MOD_TO_CLASS_IDX[mod]
            for snr in snrs:
                for _ in range(n_per):
                    sps = _SPS_BASE * (1.0 + np.random.uniform(-_SPS_ERR_MAX, _SPS_ERR_MAX))
                    iq  = add_awgn(generate_signal(mod, sps=sps), snr)
                    feat = _preprocess(np.real(iq).astype(np.float32),
                                       np.imag(iq).astype(np.float32))
                    x   = torch.from_numpy(feat).unsqueeze(0).to(device)
                    pred = int(model(x).argmax(1).item())
                    hit  = int(pred == true_label)
                    results[snr][0]           += hit
                    results[snr][1]           += 1
                    per_mod_acc[mod][snr][0]  += hit
                    per_mod_acc[mod][snr][1]  += 1
                    done += 1
            print(f"  {mod:<10} done", flush=True)

    overall_acc = {snr: results[snr][0] / results[snr][1] for snr in snrs}
    mod_acc     = {mod: {snr: per_mod_acc[mod][snr][0] / per_mod_acc[mod][snr][1]
                         for snr in snrs}
                   for mod in MODULATIONS}
    return snrs, overall_acc, mod_acc


def plot(snrs, overall_acc, mod_acc, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor('#f8f9fa')
    fig.suptitle('AMCNet SNR별 분류 정확도  (IQ 길이=2048)', fontsize=13, fontweight='bold')

    # ── (a) 전체 + 그룹별 ──────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor('#fdfdfd')

    # 그룹별 평균 정확도 선
    for grp, mods in GROUPS.items():
        grp_acc = np.array([[mod_acc[m][snr] for snr in snrs] for m in mods]).mean(axis=0)
        ax.plot(snrs, grp_acc * 100, color=GROUP_COLORS[grp],
                linewidth=1.6, linestyle='--', alpha=0.7, label=f'{grp} (avg)')

    # 전체 정확도 굵게
    ax.plot(snrs, [overall_acc[s] * 100 for s in snrs],
            color='black', linewidth=2.5, marker='o', markersize=6, label='Overall', zorder=5)

    ax.axhline(80, ls=':', color='gray', lw=1, alpha=0.6)
    ax.set_title('(a) 전체 및 변조 그룹별 정확도', fontsize=11)
    ax.set_xlabel('SNR (dB)'); ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 102); ax.set_xlim(snrs[0] - 0.5, snrs[-1] + 0.5)
    ax.legend(fontsize=9, loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.35)

    # SNR 주요 포인트 주석
    for snr in [0, 10, 20]:
        if snr in overall_acc:
            acc = overall_acc[snr] * 100
            ax.annotate(f'{acc:.1f}%', xy=(snr, acc),
                        xytext=(snr + 0.3, acc - 5),
                        fontsize=8, color='black')

    # ── (b) 변조별 전체 히트맵 ────────────────────────────────────────
    ax2 = axes[1]
    acc_mat = np.array([[mod_acc[m][snr] * 100 for snr in snrs] for m in MODULATIONS])
    im = ax2.imshow(acc_mat, aspect='auto', cmap='RdYlGn',
                    vmin=0, vmax=100, origin='upper', interpolation='nearest')
    ax2.set_xticks(range(len(snrs)))
    ax2.set_xticklabels([str(s) for s in snrs], fontsize=8)
    ax2.set_yticks(range(len(MODULATIONS)))
    ax2.set_yticklabels(MODULATIONS, fontsize=8)
    ax2.set_xlabel('SNR (dB)')
    ax2.set_title('(b) 변조 방식별 정확도 히트맵 (%)', fontsize=11)
    cb = plt.colorbar(im, ax=ax2, fraction=0.025, pad=0.02)
    cb.set_label('%', fontsize=9)

    for i, mod in enumerate(MODULATIONS):
        for j, snr in enumerate(snrs):
            val = acc_mat[i, j]
            ax2.text(j, i, f'{val:.0f}', ha='center', va='center', fontsize=6,
                     color='white' if val < 40 or val > 85 else '#111')

    # 그룹 구분선
    boundaries = []
    prev_grp = MOD_TO_GROUP.get(MODULATIONS[0])
    for i, m in enumerate(MODULATIONS[1:], 1):
        grp = MOD_TO_GROUP.get(m)
        if grp != prev_grp:
            boundaries.append(i - 0.5)
            prev_grp = grp
    for b in boundaries:
        ax2.axhline(b, color='navy', lw=1.2, ls='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"\n저장: {out_path}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--weights', default='model/best.pth')
    p.add_argument('--n',       type=int, default=200)
    args = p.parse_args()

    print(f"모델: {args.weights}  n={args.n}/변조/SNR")
    snrs, overall_acc, mod_acc = evaluate(args.weights, args.n)

    print("\n── SNR별 전체 정확도 ──")
    for snr in snrs:
        acc = overall_acc[snr]
        bar = '#' * int(acc * 30)
        print(f"  {snr:+4d} dB | {acc:.3f}  {bar}")

    out = os.path.join(os.path.dirname(os.path.abspath(args.weights)),
                       '..', 'docs', 'snr_accuracy.png')
    plot(snrs, overall_acc, mod_acc, os.path.normpath(out))
