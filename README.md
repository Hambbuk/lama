# LaMa: Large Mask Inpainting

딥러닝 기반 이미지 인페인팅 모델

## 🚀 Quick Start

### 1. 설치
```bash
pip install -r requirements.txt
```

### 2. 학습
```bash
python train.py --config configs/training/big-lama.yaml
```

### 3. 추론
```bash
python predict.py --model_path checkpoints/model.ckpt --input_dir images/ --output_dir results/
```

### 4. ONNX 변환
```bash
python export_onnx.py --config configs/training/big-lama.yaml --checkpoint checkpoints/model.ckpt --output model.onnx
```

## 📁 프로젝트 구조

```
├── train.py           # 학습 스크립트
├── predict.py         # 추론 스크립트  
├── export_onnx.py     # ONNX 변환 스크립트
├── lama/             # 핵심 소스코드
├── configs/          # 설정 파일들
├── util/             # 유틸리티 도구들
└── requirements.txt
```

## ⚙️ 설정 파일

- `configs/training/big-lama.yaml` - 메인 모델 (권장)
- `configs/training/lama-fourier.yaml` - Fourier 버전
- `configs/training/lama-regular.yaml` - Regular 버전

## 🛠️ 유틸리티

- `util/gen_mask_dataset.py` - 마스크 데이터셋 생성
- `util/make_checkpoint.py` - 체크포인트 변환

## 📋 요구사항

- Python 3.7+
- PyTorch 1.8+
- CUDA (GPU 사용시)
