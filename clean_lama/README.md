# LaMa Inpainting

깔끔하고 효율적인 LaMa (Large Mask Inpainting) 모델 구현입니다.

## 특징

- ✅ 간단한 설치 - CUDA 버전 자동 감지 및 적절한 PyTorch 설치
- ✅ 멀티 GPU 지원
- ✅ 깔끔한 코드 구조
- ✅ Train, Inference, ONNX Export 지원
- ✅ Import 오류 없는 안정적인 환경

## 설치

### 1. 환경 설정 (자동)

```bash
# Python 3.7+ 환경에서 실행
pip install -e .
```

이 명령어는 자동으로:
- 시스템의 CUDA 버전을 감지합니다
- 적절한 PyTorch 버전을 설치합니다
- 모든 필요한 패키지를 설치합니다

### 2. 수동 설치 (선택사항)

특정 환경에 맞춰 수동으로 설치하려면:

```bash
# CUDA 11.3
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# CPU only
pip install torch==1.10.0 torchvision==0.11.0

# 나머지 패키지
pip install -r requirements.txt
```

## 사용법

### 1. 학습 (Training)

```bash
# 단일 GPU
python scripts/train.py --config configs/train.yaml

# 멀티 GPU (예: 4개)
python scripts/train.py --config configs/train.yaml --gpus 4

# 추가 옵션
python scripts/train.py \
    --config configs/train.yaml \
    --gpus 2 \
    --batch-size 16 \
    --epochs 200 \
    --lr 0.0001 \
    --exp-name my_experiment
```

### 2. 추론 (Inference)

```bash
# 단일 이미지
python scripts/inference.py \
    --checkpoint experiments/my_experiment/checkpoints/last.ckpt \
    --input image.jpg \
    --mask mask.jpg \
    --output result.jpg

# 배치 처리
python scripts/inference.py \
    --checkpoint experiments/my_experiment/checkpoints/last.ckpt \
    --input-dir ./images \
    --mask-dir ./masks \
    --output-dir ./results

# GPU/CPU 선택 및 추가 옵션
python scripts/inference.py \
    --checkpoint model.ckpt \
    --input-dir ./images \
    --mask-dir ./masks \
    --output-dir ./results \
    --device cuda \
    --image-size 512 \
    --refinement \
    --fp16
```

### 3. ONNX 변환

```bash
# 기본 변환
python scripts/export_onnx.py \
    --checkpoint model.ckpt \
    --output model.onnx

# 추가 옵션
python scripts/export_onnx.py \
    --checkpoint model.ckpt \
    --output model.onnx \
    --image-size 512 \
    --opset-version 11 \
    --simplify \
    --dynamic-axes \
    --fp16 \
    --verify
```

## 데이터 준비

### 디렉토리 구조

```
data/
├── train/
│   ├── images/      # 학습 이미지
│   └── masks/       # 학습 마스크 (이미지와 동일한 파일명)
└── val/
    ├── images/      # 검증 이미지
    └── masks/       # 검증 마스크
```

### 마스크 형식
- 흑백 이미지 (검은색: 0, 흰색: 255)
- 흰색 영역이 인페인팅될 부분

## 설정 파일

`configs/train.yaml`에서 학습 설정을 조정할 수 있습니다:

```yaml
model:
  ngf: 64              # Generator 필터 수
  n_blocks: 18         # ResNet 블록 수
  ratio: 0.75          # FFC ratio

training:
  batch_size: 8        # GPU당 배치 크기
  epochs: 100          # 학습 에폭
  lr: 0.0002           # 학습률

losses:
  l1: 1.0              # L1 loss 가중치
  perceptual: 10.0     # Perceptual loss 가중치
  style: 250.0         # Style loss 가중치
  adversarial: 0.1     # Adversarial loss 가중치
```

## 프로젝트 구조

```
clean_lama/
├── scripts/           # 실행 스크립트
│   ├── train.py      # 학습
│   ├── inference.py  # 추론
│   └── export_onnx.py # ONNX 변환
├── src/              # 소스 코드
│   ├── models/       # 모델 구현
│   ├── losses/       # Loss 함수
│   ├── data/         # 데이터 로더
│   └── utils/        # 유틸리티
├── configs/          # 설정 파일
├── experiments/      # 실험 결과 (자동 생성)
└── setup.py         # 패키지 설치
```

## 성능 팁

1. **멀티 GPU 학습**: `--gpus N` 옵션 사용
2. **Mixed Precision**: `--fp16` 옵션으로 메모리 사용량 감소 및 속도 향상
3. **배치 크기**: GPU 메모리에 맞춰 조정 (V100: 16-32, RTX 3090: 8-16)
4. **이미지 크기**: 512x512 권장, 더 큰 이미지는 메모리 사용량 증가

## 문제 해결

### CUDA 관련 오류
```bash
# CUDA 버전 확인
nvidia-smi

# PyTorch CUDA 확인
python -c "import torch; print(torch.cuda.is_available())"
```

### 메모리 부족
- 배치 크기 감소: `--batch-size 4`
- FP16 사용: `--fp16`
- 이미지 크기 감소: `--image-size 256`

## 라이센스

본 프로젝트는 회사 내부 사용을 위해 정리된 버전입니다.