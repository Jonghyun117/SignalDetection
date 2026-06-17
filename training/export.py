# training/export.py
import argparse, os, time
import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat
from model import AMCNet
from dataset import _preprocess
from simulate import generate_signal, add_awgn, MODULATIONS, CLASSES


class _ModelWithSoftmax(nn.Module):
    """Wraps AMCNet to emit softmax probabilities in the ONNX graph."""
    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, x):
        return torch.softmax(self.base(x), dim=1)


class _CalibReader(CalibrationDataReader):
    def __init__(self, n_per_class=30):
        snrs  = [-10, -5, 0, 5, 10, 15, 20]
        items = []
        for mod in MODULATIONS:
            for snr in snrs:
                for _ in range(n_per_class):
                    iq   = add_awgn(generate_signal(mod), snr)
                    feat = _preprocess(np.real(iq).astype(np.float32),
                                       np.imag(iq).astype(np.float32))
                    items.append({'input': feat[np.newaxis].astype(np.float32)})
        self._iter = iter(items)

    def get_next(self):
        return next(self._iter, None)


def export(args):
    model = AMCNet(num_classes=len(CLASSES))
    model.load_state_dict(torch.load(args.weights, map_location='cpu'))
    model.eval()
    wrapped = _ModelWithSoftmax(model)
    wrapped.eval()

    fp32_path = args.output.replace('.onnx', '_fp32.onnx')
    dummy = torch.randn(1, 3, 1024)
    torch.onnx.export(
        wrapped, dummy, fp32_path,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        opset_version=18,
    )
    onnx.checker.check_model(fp32_path)
    print(f"FP32 ONNX -> {fp32_path}")

    quantize_static(
        fp32_path, args.output,
        calibration_data_reader=_CalibReader(n_per_class=30),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
    )
    print(f"INT8 ONNX -> {args.output}")

    # 현재 머신 레이턴시 참고용
    sess = ort.InferenceSession(args.output)
    x    = np.random.randn(1, 3, 1024).astype(np.float32)
    sess.run(None, {'input': x})       # warm-up
    t0 = time.perf_counter()
    for _ in range(100):
        sess.run(None, {'input': x})
    ms = (time.perf_counter() - t0) / 100 * 1000
    print(f"INT8 latency (this machine): {ms:.2f} ms/sample")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--weights', default='model/best.pth')
    p.add_argument('--output',  default='model/amc_model.onnx')
    args = p.parse_args()
    os.makedirs('model', exist_ok=True)
    export(args)
