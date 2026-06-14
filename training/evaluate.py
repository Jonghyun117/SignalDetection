# training/evaluate.py
import argparse
import numpy as np
import onnxruntime as ort
from simulate import generate_signal, add_awgn, MODULATIONS
from dataset import _preprocess


def evaluate(model_path, n_per_class=200):
    sess = ort.InferenceSession(model_path)
    snrs = list(range(-10, 21, 2))
    results = {snr: [0, 0] for snr in snrs}   # [correct, total]

    for label, mod in enumerate(MODULATIONS):
        for snr in snrs:
            for _ in range(n_per_class):
                iq   = add_awgn(generate_signal(mod), snr)
                feat = _preprocess(np.real(iq).astype(np.float32),
                                   np.imag(iq).astype(np.float32))
                out  = sess.run(None, {'input': feat[np.newaxis].astype(np.float32)})[0]
                results[snr][0] += int(np.argmax(out) == label)
                results[snr][1] += 1

    print(f"\n{'SNR(dB)':>8} {'Accuracy':>10}")
    print('-' * 20)
    for snr in snrs:
        acc = results[snr][0] / results[snr][1]
        bar = '#' * int(acc * 20)
        print(f"{snr:>8}   {acc:.3f}  {bar}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='model/amc_model.onnx')
    p.add_argument('--n',     type=int, default=200)
    args = p.parse_args()
    evaluate(args.model, args.n)
