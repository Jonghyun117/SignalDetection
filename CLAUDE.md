# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**SignalDetection** — ZCU208-1 보드 PS(ARM Cortex-A53)에서 동작하는 자동 변조 방식 분류(AMC) 시스템.  
PL에서 생성된 길이 2048 IQ 데이터를 받아 17개 변조 방식으로 분류한다.

## Commands

### Python (학습 파이프라인, 별도 PC)

```bash
# 환경 설치
pip install -r training/requirements.txt

# 학습
python training/train.py --epochs 50 --n_per 500 --batch 256 --save model/best.pth

# ONNX INT8 변환
python training/export.py --weights model/best.pth --output model/amc_model.onnx

# SNR별 정확도 평가
python training/evaluate.py --model model/amc_model.onnx --n 200

# 단위 테스트
python -m pytest tests/test_simulate.py tests/test_model.py tests/test_dataset.py -v
```

### C (추론 파이프라인, ZCU208-1)

```bash
# 빌드 (ORT_ROOT = ONNX Runtime ARM64 압축 해제 경로)
mkdir -p inference/build && cd inference/build
cmake .. -DORT_ROOT=/path/to/onnxruntime
make -j4

# 단위 테스트
ctest --output-on-failure

# 전처리 테스트만
gcc -o tests/test_preprocess tests/test_preprocess.c inference/preprocess.c \
    -I inference/ -lm -std=c11
./tests/test_preprocess

# 실행 (stdin: float32 I[2048] + Q[2048] binary)
./inference/build/amc_infer model/amc_model.onnx
```

## Architecture

두 개의 독립 파이프라인으로 구성된다.

**Python 학습 파이프라인** (`training/`):
- `simulate.py` → `dataset.py` → `model.py` → `train.py` → `export.py`
- `simulate.py`의 `generate_signal()`이 17종 변조 IQ를 생성, `add_awgn()`으로 SNR 제어
- `dataset.py`의 `_preprocess(i, q)` → `(3, 2048)` 순시 특징(진폭/위상/주파수) 변환. **C 전처리와 동일 로직 유지 필수**
- `export.py`는 모델을 `_ModelWithSoftmax`로 래핑해 ONNX에 Softmax 포함

**C 추론 파이프라인** (`inference/`):
- `preprocess.c` → `classifier.c` → `main.c` 순서로 호출
- `preprocess.c`: dataset.py의 `_preprocess()`와 동일한 DC 제거 → 전력 정규화 → 순시 특징 계산
- `classifier.c`: ONNX Runtime C API 래퍼. `amc_classifier_init()` → `amc_classifier_run()` → `amc_classifier_destroy()`
- ONNX Runtime ARM64: https://github.com/microsoft/onnxruntime/releases

**도메인 적응**: `train.py`의 `adapt_bn(model_path, real_iq_list, output_path)`로 실측 SG 데이터에서 BatchNorm 통계치만 업데이트.

## Key Constraints

- IQ 입력은 PL에서 이미 DC 제거되어 올 수 있으나, 전처리에서 한 번 더 수행함 (안전 마진)
- `preprocess.c`와 `dataset.py`의 `_preprocess()`는 반드시 동일한 수학적 동작을 해야 함
- ONNX 모델 출력 이름은 `"input"` / `"output"` (변경 시 `classifier.c`도 수정)
- 클래스 순서는 `simulate.py`의 `MODULATIONS` 리스트가 정의 (총 17개)
