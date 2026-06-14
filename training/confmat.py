# training/confmat.py — confusion matrix + grouped-class analysis
import argparse
import numpy as np
import onnxruntime as ort
from simulate import generate_signal, add_awgn, MODULATIONS
from dataset import _preprocess

# ── class grouping ──────────────────────────────────────────────────
GROUPS = {
    'AM':           ['AM'],
    'FM':           ['FM'],
    'PM':           ['PM'],
    'CW':           ['CW'],
    'PSK/DBPSK':    ['2PSK', 'DBPSK'],
    'QPSK/DQPSK':   ['4PSK', 'DQPSK', 'OQPSK'],
    '8PSK':         ['8PSK'],
    'FSK2':         ['2FSK'],
    'FSK4':         ['4FSK'],
    'FSK8':         ['8FSK'],
    'QAM/APSK':     ['8QAM', '16QAM', '16APSK', '32APSK', '64APSK', '128APSK', '256APSK'],
    'OOK':          ['OOK'],
}

MOD_TO_GROUP = {}
for grp, mods in GROUPS.items():
    for m in mods:
        MOD_TO_GROUP[m] = grp
GROUP_NAMES = list(GROUPS.keys())


def build_confmat(model_path, snrs, n_per):
    sess = ort.InferenceSession(model_path)
    n = len(MODULATIONS)
    cm = np.zeros((n, n), dtype=np.int32)

    for true_label, mod in enumerate(MODULATIONS):
        for snr in snrs:
            for _ in range(n_per):
                iq   = add_awgn(generate_signal(mod), snr)
                feat = _preprocess(np.real(iq).astype(np.float32),
                                   np.imag(iq).astype(np.float32))
                out  = sess.run(None, {'input': feat[np.newaxis].astype(np.float32)})[0]
                pred_label = int(np.argmax(out))
                cm[true_label, pred_label] += 1
    return cm


def print_confmat(cm, labels, title):
    n   = len(labels)
    w   = max(len(l) for l in labels)
    col = max(w, 4)
    print(f'\n{"="*10} {title} {"="*10}')
    header = ' ' * (w + 2) + '  '.join(f'{l:>{col}}' for l in labels)
    print(header)
    for i, row_lbl in enumerate(labels):
        total  = cm[i].sum()
        values = '  '.join(
            f'\033[1m{cm[i,j]:>{col}}\033[0m' if i == j else f'{cm[i,j]:>{col}}'
            for j in range(n)
        )
        acc = cm[i, i] / total if total else 0
        print(f'{row_lbl:>{w}}  {values}   ({acc:.0%})')


def merge_cm(cm_full):
    """Collapse full 21×21 cm into grouped cm."""
    gn  = GROUP_NAMES
    ng  = len(gn)
    g2i = {g: i for i, g in enumerate(gn)}
    cm_g = np.zeros((ng, ng), dtype=np.int32)
    for ti, tm in enumerate(MODULATIONS):
        for pi, pm in enumerate(MODULATIONS):
            tg = g2i[MOD_TO_GROUP[tm]]
            pg = g2i[MOD_TO_GROUP[pm]]
            cm_g[tg, pg] += cm_full[ti, pi]
    return cm_g


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='../model/amc_model.onnx')
    p.add_argument('--snrs', nargs='+', type=int, default=[10, 15, 20])
    p.add_argument('--n',    type=int, default=50)
    args = p.parse_args()

    print(f'SNR={args.snrs}, n={args.n}/class/SNR → {args.n*len(args.snrs)} samples/class')
    cm_full = build_confmat(args.model, args.snrs, args.n)

    print_confmat(cm_full, MODULATIONS, '21-class (full)')

    cm_grp = merge_cm(cm_full)
    print_confmat(cm_grp, GROUP_NAMES, 'Grouped classes')

    # per-group accuracy
    print('\n── Grouped accuracy ──')
    for i, g in enumerate(GROUP_NAMES):
        total = cm_grp[i].sum()
        acc   = cm_grp[i, i] / total if total else 0
        print(f'  {g:>14}: {acc:.1%}')
    total_correct = np.trace(cm_grp)
    total_all     = cm_grp.sum()
    print(f'\n  Overall grouped acc: {total_correct/total_all:.1%}')
