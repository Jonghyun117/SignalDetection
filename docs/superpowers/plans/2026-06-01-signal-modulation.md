# Signal Modulation Classification — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** ZCU208-1 PS에서 1024 IQ 샘플을 받아 17개 변조 방식을 수십 ms 이내에 분류하는 C 추론 파이프라인과 PyTorch 기반 학습 파이프라인을 구축한다.

**Architecture:** Python 학습 파이프라인(시뮬레이션 데이터 생성 → 1D-CNN 학습 → ONNX INT8 PTQ 변환)과 C 추론 파이프라인(전처리 → ONNX Runtime C API 추론)으로 구성된다. 전처리 단계에서 IQ를 순시 진폭/위상/주파수로 변환해 도메인 갭을 최소화하고, 소량의 실측 데이터로 BatchNorm 통계치만 보정해 도메인 적응을 수행한다.

**Tech Stack:** Python 3.10+, PyTorch 2.x, ONNX 1.14+, ONNX Runtime 1.17+, C11, CMake 3.20+

---

## File Map

```
SignalDetection/
├── training/
│   ├── requirements.txt  # Python 의존성
│   ├── simulate.py       # 17종 변조 IQ 데이터 생성
│   ├── dataset.py        # PyTorch Dataset + 실시간 RF 증강
│   ├── model.py          # AMCNet 1D-CNN 정의
│   ├── train.py          # 학습 루프 + BN 도메인 적응
│   ├── export.py         # ONNX 변환 + INT8 PTQ
│   └── evaluate.py       # SNR별 정확도 평가
├── tests/
│   ├── test_simulate.py  # 신호 생성 단위 테스트
│   ├── test_model.py     # 모델 forward pass 테스트
│   ├── test_preprocess.c # 전처리 C 단위 테스트
│   └── test_classifier.c # 분류기 C 단위 테스트
├── inference/
│   ├── preprocess.h      # 전처리 API 선언
│   ├── preprocess.c      # DC 제거, 정규화, 순시 특징 계산
│   ├── classifier.h      # 분류기 API 선언
│   ├── classifier.c      # ONNX Runtime C API 래퍼
│   ├── main.c            # PL→PS 인터페이스 + 분류 결과 출력
│   └── CMakeLists.txt    # 빌드 시스템
└── model/
    └── amc_model.onnx    # INT8 양자화 모델 (export.py 출력)
```

---

### Task 1: Python 환경 설정

**Files:**
- Create: `training/requirements.txt`

- [ ] **Step 1: requirements.txt 작성**

```
torch>=2.0.0
onnx>=1.14.0
onnxruntime>=1.17.0
onnxruntime-extensions>=0.8.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
```

- [ ] **Step 2: 가상환경 생성 및 패키지 설치**

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r training/requirements.txt
```

Expected: 오류 없이 설치 완료

- [ ] **Step 3: 설치 확인**

```bash
python -c "import torch; import onnx; import onnxruntime; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git init
git add training/requirements.txt
git commit -m "chore: add Python training dependencies"
```

---

### Task 2: 신호 시뮬레이션 (simulate.py)

**Files:**
- Create: `training/simulate.py`
- Create: `tests/test_simulate.py`

17개 변조 방식의 baseband IQ 신호를 생성한다. 각 신호는 `generate_signal(mod, n_samples=1024, sps=8)` 으로 호출한다. `sps`(samples per symbol) = 8 고정.

- [ ] **Step 1: 테스트 작성**

```python
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
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
python -m pytest tests/test_simulate.py -v
```

Expected: `ImportError` 또는 `ModuleNotFoundError`

- [ ] **Step 3: simulate.py 구현**

```python
# training/simulate.py
import numpy as np

MODULATIONS = [
    'AM', 'FM', 'PM', 'CW',
    '2PSK', '4PSK', '8PSK',
    '2FSK', '4FSK', '8FSK',
    '8QAM', '16QAM',
    '16APSK', '32APSK', '64APSK', '128APSK',
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
```

- [ ] **Step 4: 테스트 실행**

```bash
python -m pytest tests/test_simulate.py -v
```

Expected: 5개 PASS

- [ ] **Step 5: Commit**

```bash
git add training/simulate.py tests/test_simulate.py
git commit -m "feat: add IQ signal simulation for 17 modulation types"
```

---

### Task 3: PyTorch Dataset + 증강 (dataset.py)

**Files:**
- Create: `training/dataset.py`
- Create: `tests/test_dataset.py`

전처리(DC 제거 → 정규화 → 순시 특징)와 RF 열화 증강을 포함한다. `_preprocess` 함수는 C 전처리와 동일한 로직을 Python으로 구현해야 한다.

- [ ] **Step 1: 테스트 작성**

```python
# tests/test_dataset.py
import torch
import numpy as np
import sys
sys.path.insert(0, 'training')
from dataset import AMCDataset, _preprocess
from simulate import MODULATIONS

def test_dataset_output_shape():
    ds = AMCDataset(n_per_class_per_snr=2, augment=False)
    x, label = ds[0]
    assert x.shape == (3, 1024), f"Expected (3, 1024), got {x.shape}"
    assert 0 <= label < len(MODULATIONS)

def test_preprocess_power_normalized():
    i_arr = np.random.randn(1024).astype(np.float32) * 5.0
    q_arr = np.random.randn(1024).astype(np.float32) * 5.0
    feat = _preprocess(i_arr, q_arr)
    # amplitude channel: mean(A²) should be ~1
    mean_power = np.mean(feat[0] ** 2)
    assert abs(mean_power - 1.0) < 0.05, f"mean power={mean_power:.4f}"

def test_augmentation_changes_signal():
    ds_clean = AMCDataset(n_per_class_per_snr=2, augment=False)
    ds_aug   = AMCDataset(n_per_class_per_snr=2, augment=True)
    x_clean, _ = ds_clean[0]
    x_aug,   _ = ds_aug[0]
    assert not torch.allclose(x_clean, x_aug), "Augmented signal should differ"
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
python -m pytest tests/test_dataset.py -v
```

Expected: `ImportError`

- [ ] **Step 3: dataset.py 구현**

```python
# training/dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset
from simulate import generate_signal, add_awgn, MODULATIONS


def _preprocess(i_arr, q_arr):
    """DC removal → power normalization → instantaneous features.
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

    # Frequency offset ±5% of normalized fs=1
    f_off  = np.random.uniform(-0.05, 0.05)
    iq     = (i_arr + 1j * q_arr) * np.exp(1j * 2 * np.pi * f_off * t)
    i_arr, q_arr = np.real(iq).astype(np.float32), np.imag(iq).astype(np.float32)

    # Phase offset uniform [0, 2π]
    phi_off = np.random.uniform(0, 2 * np.pi)
    iq      = (i_arr + 1j * q_arr) * np.exp(1j * phi_off)
    i_arr, q_arr = np.real(iq).astype(np.float32), np.imag(iq).astype(np.float32)

    # IQ amplitude imbalance ±1 dB
    amp_imb = 10 ** (np.random.uniform(-1.0, 1.0) / 20.0)
    i_arr  *= amp_imb

    # IQ phase imbalance ±5°
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
```

- [ ] **Step 4: 테스트 실행**

```bash
python -m pytest tests/test_dataset.py -v
```

Expected: 3개 PASS

- [ ] **Step 5: Commit**

```bash
git add training/dataset.py tests/test_dataset.py
git commit -m "feat: add AMCDataset with RF augmentation and preprocessing"
```

---

### Task 4: CNN 모델 정의 (model.py)

**Files:**
- Create: `training/model.py`
- Create: `tests/test_model.py`

- [ ] **Step 1: 테스트 작성**

```python
# tests/test_model.py
import torch
import sys
sys.path.insert(0, 'training')
from model import AMCNet

def test_forward_shape():
    model = AMCNet(num_classes=17)
    out = model(torch.randn(4, 3, 1024))
    assert out.shape == (4, 17), f"Expected (4,17), got {out.shape}"

def test_param_count():
    model = AMCNet(num_classes=17)
    n = sum(p.numel() for p in model.parameters())
    assert n < 500_000, f"Too many params: {n}"

def test_eval_deterministic():
    model = AMCNet(num_classes=17)
    model.eval()
    x = torch.randn(1, 3, 1024)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    assert torch.allclose(out1, out2), "Eval mode should be deterministic"
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
python -m pytest tests/test_model.py -v
```

Expected: `ImportError`

- [ ] **Step 3: model.py 구현**

```python
# training/model.py
import torch.nn as nn


class AMCNet(nn.Module):
    def __init__(self, num_classes=17):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=7),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.pool       = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.pool(self.features(x)).squeeze(-1)
        return self.classifier(x)
```

- [ ] **Step 4: 테스트 실행**

```bash
python -m pytest tests/test_model.py -v
```

Expected: 3개 PASS

- [ ] **Step 5: Commit**

```bash
git add training/model.py tests/test_model.py
git commit -m "feat: add AMCNet 1D-CNN architecture"
```

---

### Task 5: 학습 루프 + BN 도메인 적응 (train.py)

**Files:**
- Create: `training/train.py`

- [ ] **Step 1: train.py 구현**

```python
# training/train.py
import argparse, os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from model import AMCNet
from dataset import AMCDataset, _preprocess
import numpy as np


def _eval_accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total   += len(y)
    return correct / total


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ds     = AMCDataset(n_per_class_per_snr=args.n_per, augment=True)
    n_val  = int(0.15 * len(ds))
    tr_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val],
                                 generator=torch.Generator().manual_seed(42))
    tr_loader  = DataLoader(tr_ds,  batch_size=args.batch, shuffle=True,  num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2)

    model     = AMCNet(num_classes=17).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_val = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = correct = total = 0
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
            correct  += (logits.argmax(1) == y).sum().item()
            total    += len(y)
        scheduler.step()
        val_acc = _eval_accuracy(model, val_loader, device)
        print(f"Epoch {epoch:3d} | loss={loss_sum/total:.4f} "
              f"| train={correct/total:.3f} | val={val_acc:.3f}")
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), args.save)
    print(f"Best val_acc: {best_val:.3f} → saved to {args.save}")


def adapt_bn(model_path, real_iq_list, output_path):
    """Update only BatchNorm running statistics using real SG data.

    real_iq_list: list of (i_arr, q_arr) float32 numpy arrays from the real SG.
    All model weights are frozen; only BN running_mean/running_var are updated.
    Typically 200-500 samples per class are sufficient.
    """
    model = AMCNet(num_classes=17)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))

    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.train()
            m.reset_running_stats()
        else:
            for p in m.parameters():
                p.requires_grad_(False)

    with torch.no_grad():
        for i_arr, q_arr in real_iq_list:
            feat = _preprocess(i_arr, q_arr)
            model(torch.from_numpy(feat).unsqueeze(0))

    torch.save(model.state_dict(), output_path)
    print(f"BN-adapted model saved → {output_path}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int,   default=50)
    p.add_argument('--batch',  type=int,   default=256)
    p.add_argument('--lr',     type=float, default=1e-3)
    p.add_argument('--n_per',  type=int,   default=500)
    p.add_argument('--save',   default='model/best.pth')
    args = p.parse_args()
    os.makedirs('model', exist_ok=True)
    train(args)
```

- [ ] **Step 2: Smoke test (소형 데이터셋 1 epoch)**

```bash
python training/train.py --epochs 1 --n_per 5 --batch 32 --save model/smoke.pth
```

Expected: Epoch 1 출력, 오류 없음, `model/smoke.pth` 생성

- [ ] **Step 3: Commit**

```bash
git add training/train.py
git commit -m "feat: add training loop and BN domain adaptation"
```

---

### Task 6: ONNX 변환 + INT8 PTQ (export.py)

**Files:**
- Create: `training/export.py`

모델은 logits를 출력하므로 ONNX export 시 Softmax를 래핑해서 추론 코드에서 별도 처리 없이 확률값을 얻는다.

- [ ] **Step 1: export.py 구현**

```python
# training/export.py
import argparse, os, time
import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
from model import AMCNet
from dataset import _preprocess
from simulate import generate_signal, add_awgn, MODULATIONS


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
    model = AMCNet(num_classes=17)
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
        opset_version=17,
    )
    onnx.checker.check_model(fp32_path)
    print(f"FP32 ONNX → {fp32_path}")

    quantize_static(
        fp32_path, args.output,
        calibration_data_reader=_CalibReader(n_per_class=30),
        quant_type=QuantType.QInt8,
    )
    print(f"INT8 ONNX → {args.output}")

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
```

- [ ] **Step 2: Smoke test**

```bash
python training/export.py --weights model/smoke.pth --output model/amc_model.onnx
```

Expected: `FP32 ONNX`, `INT8 ONNX` 저장 메시지, latency 출력, 오류 없음

- [ ] **Step 3: output 확인**

```bash
python -c "
import onnxruntime as ort, numpy as np
sess = ort.InferenceSession('model/amc_model.onnx')
out  = sess.run(None, {'input': np.random.randn(1,3,1024).astype(np.float32)})[0]
print('shape:', out.shape, '| sum:', out.sum())
"
```

Expected: `shape: (1, 17) | sum: ~1.0`

- [ ] **Step 4: Commit**

```bash
git add training/export.py
git commit -m "feat: add ONNX export with softmax wrapper and INT8 PTQ"
```

---

### Task 7: SNR별 정확도 평가 (evaluate.py)

**Files:**
- Create: `training/evaluate.py`

- [ ] **Step 1: evaluate.py 구현**

```python
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
```

- [ ] **Step 2: Smoke test**

```bash
python training/evaluate.py --model model/amc_model.onnx --n 10
```

Expected: SNR별 정확도 테이블 출력, 오류 없음 (smoke 모델이라 정확도는 낮음)

- [ ] **Step 3: Commit**

```bash
git add training/evaluate.py
git commit -m "feat: add SNR-wise accuracy evaluation"
```

---

### Task 8: C 전처리 (preprocess.h + preprocess.c)

**Files:**
- Create: `inference/preprocess.h`
- Create: `inference/preprocess.c`
- Create: `tests/test_preprocess.c`

`_preprocess()` in dataset.py와 동일한 로직을 C로 구현한다.

- [ ] **Step 1: 테스트 작성**

```c
/* tests/test_preprocess.c */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../inference/preprocess.h"

#define N 1024

static void fail(const char *msg, float got, float exp) {
    fprintf(stderr, "FAIL %s: expected %.6f, got %.6f\n", msg, exp, got);
    exit(1);
}

static void test_dc_removal(void) {
    float i_in[N], q_in[N], feat[3][N];
    /* Alternating signal with large DC offset */
    for (int t = 0; t < N; t++) {
        i_in[t] = 10.0f + (t % 2 == 0 ? 1.0f : -1.0f);
        q_in[t] = 5.0f  + (t % 2 == 0 ? 1.0f : -1.0f);
    }
    amc_preprocess(i_in, q_in, feat, N);
    float mean_i = 0.0f;
    for (int t = 0; t < N; t++) mean_i += i_in[t] - 10.0f; /* approx zero mean */
    /* After preprocessing, amplitude channel mean should reflect unit power, not DC */
    float mean_a = 0.0f;
    for (int t = 0; t < N; t++) mean_a += feat[0][t];
    mean_a /= N;
    if (fabsf(mean_a - 1.0f) > 0.01f)
        fail("dc_removal: mean amplitude", mean_a, 1.0f);
    printf("PASS: test_dc_removal\n");
}

static void test_power_normalization(void) {
    float i_in[N], q_in[N], feat[3][N];
    for (int t = 0; t < N; t++) {
        i_in[t] = (t % 2 == 0) ?  3.0f : -3.0f;
        q_in[t] = (t % 2 == 0) ?  4.0f : -4.0f;
    }
    amc_preprocess(i_in, q_in, feat, N);
    float mean_pow = 0.0f;
    for (int t = 0; t < N; t++) mean_pow += feat[0][t] * feat[0][t];
    mean_pow /= N;
    if (fabsf(mean_pow - 1.0f) > 0.02f)
        fail("power_normalization: mean(A²)", mean_pow, 1.0f);
    printf("PASS: test_power_normalization\n");
}

static void test_bpsk_phase(void) {
    /* BPSK: alternating ±1 on I, Q=0 → phase alternates between 0 and ±π */
    float i_in[N], q_in[N], feat[3][N];
    for (int t = 0; t < N; t++) {
        i_in[t] = (t % 16 < 8) ? 1.0f : -1.0f;
        q_in[t] = 0.0f;
    }
    amc_preprocess(i_in, q_in, feat, N);
    for (int t = 0; t < N; t++) {
        float phi = feat[1][t];
        if (fabsf(phi) > 0.1f && fabsf(fabsf(phi) - 3.14159f) > 0.1f) {
            fprintf(stderr, "FAIL bpsk_phase: phi[%d]=%.4f\n", t, phi);
            exit(1);
        }
    }
    printf("PASS: test_bpsk_phase\n");
}

int main(void) {
    test_dc_removal();
    test_power_normalization();
    test_bpsk_phase();
    printf("All preprocess tests passed.\n");
    return 0;
}
```

- [ ] **Step 2: preprocess.h 구현**

```c
/* inference/preprocess.h */
#ifndef AMC_PREPROCESS_H
#define AMC_PREPROCESS_H

/* Preprocess 1024-sample IQ into 3-channel instantaneous features.
   features[0]: amplitude |A(t)|
   features[1]: instantaneous phase φ(t)
   features[2]: instantaneous frequency Δφ(t) (phase increment, unwrapped)
   n must equal 1024. */
void amc_preprocess(const float *i_samples, const float *q_samples,
                    float features[3][1024], int n);

#endif
```

- [ ] **Step 3: preprocess.c 구현**

```c
/* inference/preprocess.c */
#include "preprocess.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

void amc_preprocess(const float *i_in, const float *q_in,
                    float features[3][1024], int n)
{
    float i_buf[1024], q_buf[1024], phi[1024];
    int t;

    /* DC removal */
    float mi = 0.0f, mq = 0.0f;
    for (t = 0; t < n; t++) { mi += i_in[t]; mq += q_in[t]; }
    mi /= n; mq /= n;
    for (t = 0; t < n; t++) { i_buf[t] = i_in[t] - mi; q_buf[t] = q_in[t] - mq; }

    /* Power normalization: mean(I²+Q²) = 1 */
    float power = 0.0f;
    for (t = 0; t < n; t++) power += i_buf[t]*i_buf[t] + q_buf[t]*q_buf[t];
    power /= n;
    float scale = (power > 1e-10f) ? 1.0f / sqrtf(power) : 1.0f;
    for (t = 0; t < n; t++) { i_buf[t] *= scale; q_buf[t] *= scale; }

    /* Instantaneous amplitude */
    for (t = 0; t < n; t++)
        features[0][t] = sqrtf(i_buf[t]*i_buf[t] + q_buf[t]*q_buf[t]);

    /* Instantaneous phase */
    for (t = 0; t < n; t++) {
        phi[t]        = atan2f(q_buf[t], i_buf[t]);
        features[1][t] = phi[t];
    }

    /* Instantaneous frequency: unwrapped phase increment */
    features[2][0] = 0.0f;
    for (t = 1; t < n; t++) {
        float d = phi[t] - phi[t-1];
        while (d >  (float)M_PI) d -= 2.0f * (float)M_PI;
        while (d < -(float)M_PI) d += 2.0f * (float)M_PI;
        features[2][t] = d;
    }
}
```

- [ ] **Step 4: 테스트 컴파일 및 실행**

```bash
gcc -o tests/test_preprocess \
    tests/test_preprocess.c inference/preprocess.c \
    -I inference/ -lm -std=c11
./tests/test_preprocess
```

Expected:
```
PASS: test_dc_removal
PASS: test_power_normalization
PASS: test_bpsk_phase
All preprocess tests passed.
```

- [ ] **Step 5: Commit**

```bash
git add inference/preprocess.h inference/preprocess.c tests/test_preprocess.c
git commit -m "feat: add C preprocessing (DC removal, normalization, instantaneous features)"
```

---

### Task 9: ONNX Runtime C API 래퍼 (classifier.h + classifier.c)

**Files:**
- Create: `inference/classifier.h`
- Create: `inference/classifier.c`
- Create: `tests/test_classifier.c`

ONNX Runtime 1.17+ C API 사용. ZCU208-1에는 ONNX Runtime ARM64 바이너리를 별도 설치해야 한다.  
설치: https://github.com/microsoft/onnxruntime/releases → `onnxruntime-linux-aarch64-*.tgz` 다운로드 후 압축 해제.

- [ ] **Step 1: classifier.h 구현**

```c
/* inference/classifier.h */
#ifndef AMC_CLASSIFIER_H
#define AMC_CLASSIFIER_H

#define AMC_NUM_CLASSES 17

/* Class names in MODULATIONS order (matches training/simulate.py). */
extern const char *AMC_CLASS_NAMES[AMC_NUM_CLASSES];

/* Load model. Returns 0 on success, -1 on failure. */
int  amc_classifier_init(const char *model_path);

/* Run inference. features[3][1024]: preprocessed input.
   probs[17]: softmax output. Returns predicted class index, or -1 on error. */
int  amc_classifier_run(float features[3][1024], float probs[AMC_NUM_CLASSES]);

/* Release ONNX Runtime resources. */
void amc_classifier_destroy(void);

#endif
```

- [ ] **Step 2: classifier.c 구현**

```c
/* inference/classifier.c */
#include "classifier.h"
#include <stdio.h>
#include <string.h>
#include "onnxruntime_c_api.h"

const char *AMC_CLASS_NAMES[AMC_NUM_CLASSES] = {
    "AM","FM","PM","CW",
    "2PSK","4PSK","8PSK",
    "2FSK","4FSK","8FSK",
    "8QAM","16QAM",
    "16APSK","32APSK","64APSK","128APSK",
    "OOK"
};

static const OrtApi    *g_ort  = NULL;
static OrtEnv          *g_env  = NULL;
static OrtSession      *g_sess = NULL;
static OrtMemoryInfo   *g_mem  = NULL;

int amc_classifier_init(const char *model_path)
{
    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!g_ort) return -1;

    OrtStatus *st;
    if ((st = g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "amc", &g_env))) {
        fprintf(stderr, "ORT CreateEnv: %s\n", g_ort->GetErrorMessage(st)); return -1;
    }

    OrtSessionOptions *opts;
    g_ort->CreateSessionOptions(&opts);
    g_ort->SetIntraOpNumThreads(opts, 1);
    g_ort->SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_ALL);

    if ((st = g_ort->CreateSession(g_env, model_path, opts, &g_sess))) {
        fprintf(stderr, "ORT CreateSession: %s\n", g_ort->GetErrorMessage(st));
        g_ort->ReleaseSessionOptions(opts); return -1;
    }
    g_ort->ReleaseSessionOptions(opts);
    g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &g_mem);
    return 0;
}

int amc_classifier_run(float features[3][1024], float probs[AMC_NUM_CLASSES])
{
    int64_t    shape[] = {1, 3, 1024};
    OrtValue  *in_val  = NULL, *out_val = NULL;
    OrtStatus *st;

    st = g_ort->CreateTensorWithDataAsOrtValue(
            g_mem, features, 3*1024*sizeof(float),
            shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in_val);
    if (st) return -1;

    const char *in_names[]  = {"input"};
    const char *out_names[] = {"output"};
    st = g_ort->Run(g_sess, NULL,
                    in_names,  (const OrtValue *const *)&in_val,  1,
                    out_names, 1, &out_val);
    g_ort->ReleaseValue(in_val);
    if (st) { fprintf(stderr, "ORT Run: %s\n", g_ort->GetErrorMessage(st)); return -1; }

    float *data;
    g_ort->GetTensorMutableData(out_val, (void **)&data);
    memcpy(probs, data, AMC_NUM_CLASSES * sizeof(float));
    g_ort->ReleaseValue(out_val);

    int best = 0;
    for (int i = 1; i < AMC_NUM_CLASSES; i++)
        if (probs[i] > probs[best]) best = i;
    return best;
}

void amc_classifier_destroy(void)
{
    if (g_mem)  { g_ort->ReleaseMemoryInfo(g_mem);  g_mem  = NULL; }
    if (g_sess) { g_ort->ReleaseSession(g_sess);     g_sess = NULL; }
    if (g_env)  { g_ort->ReleaseEnv(g_env);          g_env  = NULL; }
}
```

- [ ] **Step 3: test_classifier.c 구현**

```c
/* tests/test_classifier.c */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../inference/classifier.h"

#ifndef MODEL_PATH
#define MODEL_PATH "model/amc_model.onnx"
#endif

static void test_init_and_destroy(void) {
    if (amc_classifier_init(MODEL_PATH) != 0) {
        fprintf(stderr, "FAIL test_init_and_destroy\n"); exit(1);
    }
    amc_classifier_destroy();
    printf("PASS: test_init_and_destroy\n");
}

static void test_output_is_probability(void) {
    amc_classifier_init(MODEL_PATH);

    float features[3][1024] = {{0}};
    for (int t = 0; t < 1024; t++) features[0][t] = 1.0f; /* unit amplitude */

    float probs[AMC_NUM_CLASSES];
    int pred = amc_classifier_run(features, probs);

    if (pred < 0 || pred >= AMC_NUM_CLASSES) {
        fprintf(stderr, "FAIL: pred=%d out of range\n", pred); exit(1);
    }
    float sum = 0.0f;
    for (int i = 0; i < AMC_NUM_CLASSES; i++) sum += probs[i];
    if (fabsf(sum - 1.0f) > 0.01f) {
        fprintf(stderr, "FAIL: probs sum=%.4f\n", sum); exit(1);
    }
    printf("PASS: test_output_is_probability (pred=%s, conf=%.3f)\n",
           AMC_CLASS_NAMES[pred], probs[pred]);
    amc_classifier_destroy();
}

int main(void) {
    test_init_and_destroy();
    test_output_is_probability();
    printf("All classifier tests passed.\n");
    return 0;
}
```

- [ ] **Step 4: 테스트 컴파일 및 실행**

`ORT_ROOT`를 ONNX Runtime 압축 해제 경로로 교체한다.

```bash
ORT_ROOT=/path/to/onnxruntime
gcc -o tests/test_classifier \
    tests/test_classifier.c inference/classifier.c \
    -I$ORT_ROOT/include -I inference/ \
    -L$ORT_ROOT/lib -lonnxruntime -lm -std=c11 \
    -Wl,-rpath,$ORT_ROOT/lib
./tests/test_classifier
```

Expected:
```
PASS: test_init_and_destroy
PASS: test_output_is_probability (pred=<class>, conf=X.XXX)
All classifier tests passed.
```

- [ ] **Step 5: Commit**

```bash
git add inference/classifier.h inference/classifier.c tests/test_classifier.c
git commit -m "feat: add ONNX Runtime C API wrapper for AMC inference"
```

---

### Task 10: 메인 통합 + 빌드 시스템 (main.c + CMakeLists.txt)

**Files:**
- Create: `inference/main.c`
- Create: `inference/CMakeLists.txt`

`main.c`는 stdin에서 바이너리 IQ를 읽는다. 실제 보드 배포 시 이 부분만 DMA/공유메모리 읽기로 교체하면 된다.

- [ ] **Step 1: main.c 구현**

```c
/* inference/main.c
   Usage: ./amc_infer <model.onnx>
   Input (stdin): float32 I[1024] then float32 Q[1024], binary, repeated per frame.
   Output (stdout): "<class> conf=X.XXX latency=XX.XXms\n" per frame. */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "preprocess.h"
#include "classifier.h"

#define N 1024

static double elapsed_ms(struct timespec *t0, struct timespec *t1) {
    return (t1->tv_sec - t0->tv_sec) * 1e3 + (t1->tv_nsec - t0->tv_nsec) * 1e-6;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.onnx>\n", argv[0]);
        return 1;
    }
    if (amc_classifier_init(argv[1]) != 0) {
        fprintf(stderr, "Classifier init failed\n");
        return 1;
    }

    float i_buf[N], q_buf[N], features[3][N], probs[AMC_NUM_CLASSES];

    while (fread(i_buf, sizeof(float), N, stdin) == N &&
           fread(q_buf, sizeof(float), N, stdin) == N)
    {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        amc_preprocess(i_buf, q_buf, features, N);
        int pred = amc_classifier_run(features, probs);

        clock_gettime(CLOCK_MONOTONIC, &t1);

        if (pred >= 0)
            printf("%s conf=%.3f latency=%.2fms\n",
                   AMC_CLASS_NAMES[pred], probs[pred], elapsed_ms(&t0, &t1));
    }

    amc_classifier_destroy();
    return 0;
}
```

- [ ] **Step 2: CMakeLists.txt 구현**

```cmake
# inference/CMakeLists.txt
cmake_minimum_required(VERSION 3.20)
project(amc_infer C)
set(CMAKE_C_STANDARD 11)

set(ORT_ROOT "" CACHE PATH "ONNX Runtime installation root (contains include/ and lib/)")
if(NOT ORT_ROOT)
    message(FATAL_ERROR "Specify ORT_ROOT: cmake -DORT_ROOT=/path/to/onnxruntime ..")
endif()

add_library(amc_preprocess STATIC preprocess.c)
target_include_directories(amc_preprocess PUBLIC .)
target_link_libraries(amc_preprocess m)

add_library(amc_classifier STATIC classifier.c)
target_include_directories(amc_classifier PUBLIC . ${ORT_ROOT}/include)
target_link_libraries(amc_classifier ${ORT_ROOT}/lib/libonnxruntime.so)

add_executable(amc_infer main.c)
target_link_libraries(amc_infer amc_preprocess amc_classifier)

# Tests
add_executable(test_preprocess ../tests/test_preprocess.c)
target_link_libraries(test_preprocess amc_preprocess)

add_executable(test_classifier ../tests/test_classifier.c)
target_link_libraries(test_classifier amc_classifier amc_preprocess)

enable_testing()
add_test(NAME preprocess COMMAND test_preprocess)
add_test(NAME classifier COMMAND test_classifier
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/..)
```

- [ ] **Step 3: CMake 빌드 확인**

```bash
mkdir -p inference/build && cd inference/build
cmake .. -DORT_ROOT=/path/to/onnxruntime
make -j4
```

Expected: `amc_infer`, `test_preprocess`, `test_classifier` 빌드 성공

- [ ] **Step 4: CTest 실행**

```bash
cd inference/build
ctest --output-on-failure
```

Expected: 2개 테스트 PASS

- [ ] **Step 5: 통합 Smoke test**

```bash
python3 -c "
import sys, numpy as np
sys.path.insert(0, 'training')
from simulate import generate_signal, add_awgn
iq = add_awgn(generate_signal('16QAM'), 10)
np.real(iq).astype(np.float32).tofile(sys.stdout.buffer)
np.imag(iq).astype(np.float32).tofile(sys.stdout.buffer)
" | ./inference/build/amc_infer model/amc_model.onnx
```

Expected: `16QAM conf=X.XXX latency=XX.XXms` 형태 출력 (smoke 모델이라 클래스가 다를 수 있음)

- [ ] **Step 6: Commit**

```bash
git add inference/main.c inference/CMakeLists.txt
git commit -m "feat: add main integration and CMake build system"
```

---

## 전체 실행 순서 요약

```bash
# 1. 실제 학습 (GPU 머신 권장)
python training/train.py --epochs 50 --n_per 500 --batch 256 --save model/best.pth

# 2. ONNX INT8 변환
python training/export.py --weights model/best.pth --output model/amc_model.onnx

# 3. SNR별 정확도 확인
python training/evaluate.py --model model/amc_model.onnx --n 200

# 4. (선택) 실측 SG 데이터로 BN 도메인 적응
python3 -c "
import sys, numpy as np
sys.path.insert(0, 'training')
from train import adapt_bn
# real_iq_list: [(i_arr, q_arr), ...] 실측 데이터
adapt_bn('model/best.pth', real_iq_list, 'model/best_adapted.pth')
"
python training/export.py --weights model/best_adapted.pth \
                           --output model/amc_model_adapted.onnx

# 5. ZCU208-1 배포
scp model/amc_model.onnx user@zcu208:/home/user/
scp inference/build/amc_infer user@zcu208:/home/user/
```
