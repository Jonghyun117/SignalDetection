# Signal Modulation Classification — Design Spec
**Date:** 2026-06-01  
**Status:** Approved

---

## Overview

ZCU208-1 보드의 PS(ARM Cortex-A53)에서 동작하는 자동 변조 방식 분류(AMC) 시스템.  
PL에서 생성된 길이 1024의 IQ 데이터를 입력받아 17개 변조 방식 중 하나로 분류한다.

---

## Requirements

| 항목 | 내용 |
|------|------|
| 플랫폼 | ZCU208-1 PS (ARM Cortex-A53 quad-core @ 1.2GHz) |
| 구현 언어 | C (추론), Python (학습) |
| 입력 | 1024 IQ 샘플 (PL → PS, float32) |
| 출력 클래스 | 17종: AM, FM, PM, CW, 2PSK, 4PSK, 8PSK, 2FSK, 4FSK, 8FSK, 8QAM, 16QAM, 16APSK, 32APSK, 64APSK, 128APSK, OOK |
| SNR 범위 | -10dB ~ 20dB |
| 레이턴시 | 수십 ms 이내 (준실시간) |
| 학습 데이터 | 시뮬레이션 pre-train + 실측 SG 데이터 도메인 적응 |
| 실측 데이터 규모 | 소량 (클래스당 수백~수천 샘플) |

---

## Architecture

```
[PL: RF → IQ 생성]
        ↓ 1024 IQ samples (float32)
[PS: 전처리 — preprocess.c]
  1. DC 제거
  2. 전력 정규화
  3. 순시 특징 계산 → [3][1024]
        ↓
[PS: CNN 추론 — classifier.c / ONNX Runtime C API]
  경량 1D-CNN (INT8 양자화)
        ↓
[17클래스 softmax → 분류 결과]
```

---

## 1. Preprocessing Pipeline (C)

입력: `float I[1024], Q[1024]`  
출력: `float features[3][1024]`

1. **DC 제거**  
   `I[t] -= mean(I)`, `Q[t] -= mean(Q)`

2. **전력 정규화**  
   `scale = 1 / sqrt(mean(I² + Q²))`  
   `I[t] *= scale`, `Q[t] *= scale`

3. **순시 특징 계산**
   - `A[t]    = sqrtf(I[t]² + Q[t]²)` — 진폭 포락선
   - `phi[t]  = atan2f(Q[t], I[t])` — 순시 위상
   - `dphi[t] = unwrap(phi[t+1] - phi[t])` — 순시 주파수 (위상 증분)

이 변환으로 주파수 오프셋은 `dphi`의 상수 편이로, 이득 변동은 정규화로 흡수되어 도메인 갭을 줄인다.

---

## 2. CNN Model Architecture

학습: PyTorch | 배포: ONNX (INT8 양자화) | 추론: ONNX Runtime C API

```
Input:          [3, 1024]
Conv1D(3→32,  k=7) + BN + ReLU   → [32, 1018]
MaxPool(2)                         → [32, 509]
Conv1D(32→64, k=5) + BN + ReLU   → [64, 505]
MaxPool(2)                         → [64, 252]
Conv1D(64→128, k=5) + BN + ReLU  → [128, 248]
MaxPool(2)                         → [128, 124]
Conv1D(128→128, k=3) + BN + ReLU → [128, 122]
GlobalAveragePool                  → [128]
FC(128→64) + ReLU
FC(64→17) + Softmax
```

- 파라미터: ~230K
- FLOPs: ~40M
- 목표 추론 시간: INT8 기준 10~15ms (Cortex-A53)

---

## 3. Training Strategy

### 데이터 생성
- Python (commpy / scipy / GNU Radio)으로 17클래스 × SNR 구간별 시뮬레이션
- SNR: -10 ~ 20dB, 2dB 간격으로 균등 샘플링

### 증강 (학습 시 실시간 적용)
| 종류 | 범위 |
|------|------|
| 주파수 오프셋 | ±Fs × 5% 랜덤 |
| 위상 오프셋 | 0 ~ 2π 균등 |
| IQ 진폭 불균형 | ±1dB |
| IQ 위상 불균형 | ±5° |
| 타이밍 오프셋 | ±0.5 샘플 |

### 도메인 적응 (실측 SG 데이터)
- 가중치 고정 후 실측 데이터로 **BatchNorm running mean/var만 업데이트**
- 클래스당 수백 샘플로 충분

---

## 4. Deployment Structure

```
SignalDetection/
├── inference/
│   ├── preprocess.c      # DC 제거, 정규화, 순시 특징 계산
│   ├── classifier.c      # ONNX Runtime C API 래퍼
│   └── main.c            # PL→PS 인터페이스 + 결과 출력
├── model/
│   └── amc_model.onnx    # INT8 양자화 모델
└── training/             # 별도 PC에서 실행 (Python)
    ├── simulate.py       # 학습 데이터 생성
    ├── train.py          # PyTorch 학습
    └── export.py         # ONNX 변환 + INT8 정적 양자화 (PTQ, 시뮬레이션 캘리브레이션 데이터 사용)
```

---

## 5. Performance Targets

| SNR 구간 | 목표 정확도 |
|---------|-----------|
| 10 ~ 20dB | >95% |
| 0 ~ 10dB  | >85% |
| -10 ~ 0dB | >70% |

추론 레이턴시: **10~15ms** (INT8, Cortex-A53 @ 1.2GHz)  
FP32 fallback: 20~50ms (목표 충족 불확실, 모델 경량화로 보완)
