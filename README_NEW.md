# LaMa Inpainting with PyTorch Lightning & Hydra

Lama 인페인팅 모델을 PyTorch Lightning과 Hydra를 사용하여 현대적이고 모듈화된 방식으로 재구현한 버전입니다.

## 🚀 주요 특징

- **Fourier Convolutions**: 원본 LaMa 모델의 핵심인 FFC(Fast Fourier Convolution) 구현
- **PyTorch Lightning**: 최신 PyTorch Lightning 2.x 사용으로 깔끔한 학습 코드
- **Hydra Configuration**: 실험 설정 관리가 용이한 Hydra 사용
- **모듈화된 구조**: Generator, Discriminator, Loss 함수들을 독립적으로 구성
- **다양한 마스크 타입**: Random, Irregular, Outpainting 마스크 지원
- **현대적 패키지**: 최신 torch, torchvision, albumentations 등 사용
- **TensorBoard 연동**: 실험 추적 및 시각화

## 📁 프로젝트 구조

```
├── configs/                 # Hydra 설정 파일들
│   ├── config.yaml         # 기본 설정
│   ├── model/              # 모델 설정
│   ├── data/               # 데이터 설정
│   └── logger/             # 로거 설정
├── src/                    # 소스 코드
│   ├── models/             # 모델 구현
│   │   ├── fourier_conv.py # Fourier Convolution 레이어
│   │   ├── generator.py    # Generator 모델
│   │   └── discriminator.py # Discriminator 모델
│   ├── losses/             # Loss 함수들
│   │   └── losses.py       # 각종 loss 구현
│   ├── data/               # 데이터 로더
│   │   └── dataset.py      # 데이터셋 및 DataModule
│   └── lightning_module.py # PyTorch Lightning 모듈
├── train.py                # 메인 학습 스크립트
└── requirements.txt        # 패키지 의존성
```

## 🛠 설치

### 1. 환경 설정

```bash
# 가상환경 생성 (옵션)
python -m venv lama_env
source lama_env/bin/activate  # Linux/Mac
# 또는
lama_env\Scripts\activate     # Windows

# 패키지 설치
pip install -r requirements.txt
```

### 2. 데이터 준비

```bash
# 데이터 폴더 구조
your_dataset/
├── train/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

## 🚂 학습 실행

### 기본 학습

```bash
# 기본 설정으로 학습
python train.py

# 데이터 경로 지정
python train.py data_dir=/path/to/your/dataset

# GPU 개수 지정
python train.py trainer.gpus=2

# 배치 크기 변경
python train.py data.batch_size=8
```

### 실험 설정 변경

```bash
# 다른 모델 사용
python train.py model=baseline_resnet

# 다른 데이터셋 설정
python train.py data=celeba

# 옵티마이저 변경
python train.py optimizer=adamw

# 여러 설정 동시 변경
python train.py model=lama_ffc data.batch_size=16 trainer.max_epochs=200
```

### Hydra 멀티런 (하이퍼파라미터 스윕)

```bash
# 배치 크기 스윕
python train.py -m data.batch_size=4,8,16

# 학습률 스윕
python train.py -m model.optimizer_config.generator_lr=1e-4,5e-4,1e-3

# 복합 스윕
python train.py -m data.batch_size=4,8 model.optimizer_config.generator_lr=1e-4,1e-3
```

## ⚙️ 설정 설명

### 모델 설정 (configs/model/lama_ffc.yaml)

```yaml
generator_config:
  type: ffc                 # 'ffc' 또는 'baseline'
  ngf: 64                   # Generator 기본 채널 수
  n_blocks: 9               # ResNet 블록 개수
  n_downsampling: 3         # 다운샘플링 레이어 수

discriminator_config:
  type: multiscale          # 'multiscale', 'nlayer', 'projected'
  num_D: 3                  # Multi-scale discriminator 개수
  ndf: 64                   # Discriminator 기본 채널 수

loss_config:
  l1_weight: 1.0           # L1 loss 가중치
  perceptual_weight: 1.0   # Perceptual loss 가중치
  adversarial_weight: 0.1  # Adversarial loss 가중치
```

### 데이터 설정 (configs/data/places365.yaml)

```yaml
image_size: 256              # 학습 이미지 크기
load_size: 286              # 로드 후 크기 (augmentation용)
batch_size: 4               # 배치 크기

mask_config:
  types: ['random', 'irregular']  # 마스크 타입
  random_max_size: 0.4           # 랜덤 마스크 최대 크기
  irregular_max_vertex: 12       # Irregular 마스크 vertex 수
```

## 🎯 주요 기능

### 1. Fourier Convolutions
- 원본 LaMa의 핵심인 FFC 구현
- 주파수 도메인에서의 큰 receptive field 확보
- 주기적 패턴 복원에 뛰어난 성능

### 2. 다양한 마스크 타입
- **Random Mask**: 사각형 형태의 랜덤 마스크
- **Irregular Mask**: 불규칙한 형태의 마스크 (붓 stroke 시뮬레이션)
- **Outpainting Mask**: 이미지 확장을 위한 마스크

### 3. 멀티스케일 Loss
- L1 Loss: 픽셀 단위 복원
- Perceptual Loss: LPIPS 기반 지각적 품질
- ResNet Perceptual Loss: 추가적인 지각적 loss
- Adversarial Loss: GAN 기반 진짜같은 결과
- High Frequency Loss: 세부사항 보존

### 4. 현대적 학습 기법
- Gradient Penalty (R1): Discriminator 정규화
- Feature Matching: 학습 안정성 향상
- Mixed Precision Training: 메모리 효율성
- Automatic Mixed Precision

## 📊 모니터링

### TensorBoard 사용

```bash
# TensorBoard 실행
tensorboard --logdir logs/
```

학습 중 다음을 모니터링할 수 있습니다:
- 각종 loss 값들
- 학습률 변화
- 생성된 이미지 샘플
- PSNR, SSIM 등 메트릭

TensorBoard는 기본적으로 `./logs` 폴더에 로그를 저장합니다.

## 🔧 커스터마이징

### 새로운 모델 추가

1. `src/models/` 에 새 모델 파일 생성
2. `configs/model/` 에 설정 파일 추가
3. `src/models/generator.py` 의 `get_generator()` 함수에 등록

### 새로운 Loss 추가

1. `src/losses/losses.py` 에 새 loss 클래스 구현
2. `InpaintingLoss` 클래스에 통합
3. 설정 파일에서 가중치 조정

### 새로운 데이터셋 추가

1. `configs/data/` 에 새 데이터셋 설정 파일 생성
2. 필요시 `src/data/dataset.py` 의 `InpaintingDataset` 클래스 수정

## 🐛 문제 해결

### 일반적인 문제들

1. **CUDA 메모리 부족**
   ```bash
   python train.py data.batch_size=2 trainer.precision=16
   ```

2. **데이터 로딩 속도 느림**
   ```bash
   python train.py data.num_workers=8
   ```

3. **학습 불안정**
   ```bash
   python train.py model.loss_config.adversarial_weight=0.05
   ```

### 디버깅 모드

```bash
# 빠른 실행 (1 배치만)
python train.py trainer.fast_dev_run=true

# 작은 데이터셋으로 오버피팅 테스트
python train.py trainer.overfit_batches=10
```

## 📈 성능 최적화

### GPU 최적화
- Mixed precision training 사용: `trainer.precision=16`
- Gradient accumulation: `trainer.accumulate_grad_batches=4`
- DataLoader 최적화: `data.num_workers=4`, `data.pin_memory=true`

### 메모리 최적화
- 배치 크기 조정: `data.batch_size=2`
- Gradient checkpointing 사용 (필요시 구현)

## 📝 라이센스

이 프로젝트는 Apache 2.0 라이센스 하에 배포됩니다.

## 🙏 감사의 말

원본 LaMa 논문과 구현에 감사드립니다:
- [LaMa: Resolution-robust Large Mask Inpainting with Fourier Convolutions](https://arxiv.org/abs/2109.07161)
- [Original Implementation](https://github.com/saic-mdal/lama)

## 📧 문의

질문이나 이슈가 있으시면 GitHub Issues를 통해 문의해주세요.