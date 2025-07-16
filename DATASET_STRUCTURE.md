# LaMa 데이터셋 구조 가이드

## 1. 훈련 데이터셋 (Training Dataset)

### 기본 구조
```
dataset/
├── train/
│   ├── image1.jpg
│   ├── image2.jpg
│   ├── image3.jpg
│   └── ...
└── val/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

### 특징
- **이미지만 필요**: 마스크는 훈련 중에 자동으로 랜덤 생성됩니다
- **지원 포맷**: `.jpg`, `.jpeg`, `.png`
- **하위 폴더**: 서브 디렉토리 구조도 지원 (재귀적 검색)

### 예시
```
my_dataset/
├── train/
│   ├── category1/
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   └── category2/
│       ├── img001.jpg
│       └── ...
└── val/
    ├── val_img001.jpg
    ├── val_img002.jpg
    └── ...
```

## 2. 추론 데이터셋 (Inference Dataset)

### 방법 1: 같은 폴더에 이미지와 마스크
```
inference_data/
├── image1.jpg
├── image1_mask.png
├── image2.jpg
├── image2_mask.png
├── image3.jpg
├── image3_mask.png
└── ...
```

### 특징
- **이미지**: `.jpg`, `.jpeg`, `.png` 등
- **마스크**: `{이미지명}_mask.png` 형태
- **마스크 포맷**: 흑백 이미지 (255: 인페인팅할 영역, 0: 보존할 영역)

### 방법 2: 분리된 폴더 구조
```
inference_data/
├── img/
│   ├── image1.png
│   ├── image2.png
│   └── ...
└── mask/
    ├── image1_mask.png
    ├── image2_mask.png
    └── ...
```

### 특징
- **img/**: 원본 이미지들
- **mask/**: 해당하는 마스크들
- **파일명**: 마스크 파일명에서 `_mask` 또는 `-mask` 부분을 제거한 이름이 이미지 파일명과 일치해야 함

## 3. 마스크 생성 도구

### 수동 마스크 생성
```python
import cv2
import numpy as np

# 흑백 마스크 생성
mask = np.zeros((height, width), dtype=np.uint8)
# 인페인팅할 영역을 흰색(255)으로 채움
cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)  # 사각형 영역
cv2.circle(mask, (center_x, center_y), radius, 255, -1)  # 원형 영역
cv2.imwrite('image_mask.png', mask)
```

### 자동 마스크 생성 (랜덤)
```bash
# 기존 이미지에서 랜덤 마스크 생성
python -c "
import glob
import os
import cv2
import numpy as np

img_files = glob.glob('images/*.jpg')
os.makedirs('masks', exist_ok=True)

for img_file in img_files:
    img = cv2.imread(img_file)
    h, w = img.shape[:2]
    
    # 랜덤 마스크 생성
    mask = np.zeros((h, w), dtype=np.uint8)
    # 랜덤 사각형 영역
    x1, y1 = np.random.randint(0, w//2), np.random.randint(0, h//2)
    x2, y2 = np.random.randint(w//2, w), np.random.randint(h//2, h)
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    
    # 마스크 저장
    base_name = os.path.splitext(os.path.basename(img_file))[0]
    cv2.imwrite(f'masks/{base_name}_mask.png', mask)
"
```

## 4. 실제 사용 예시

### 훈련 시
```bash
python train.py \
    --config configs/training/default.yaml \
    --data_path /path/to/my_dataset \
    --gpus 2 \
    --batch_size 8
```

데이터 경로는 다음과 같이 구성:
```
/path/to/my_dataset/
├── train/          # 훈련 이미지들
└── val/            # 검증 이미지들
```

### 추론 시
```bash
python inference.py \
    --checkpoint ./outputs/checkpoints/last.ckpt \
    --input /path/to/inference_data \
    --output /path/to/results
```

데이터 경로는 다음과 같이 구성:
```
/path/to/inference_data/
├── image1.jpg
├── image1_mask.png
├── image2.jpg
├── image2_mask.png
└── ...
```

## 5. 데이터 품질 권장사항

### 이미지 요구사항
- **해상도**: 최소 256x256, 권장 512x512 이상
- **포맷**: RGB 컬러 이미지
- **품질**: 고해상도, 선명한 이미지 선호

### 마스크 요구사항
- **포맷**: 8-bit 그레이스케일 PNG
- **값**: 0 (보존), 255 (인페인팅)
- **크기**: 원본 이미지와 동일한 크기

### 훈련 데이터 권장사항
- **수량**: 최소 10,000장 이상 권장
- **다양성**: 다양한 장면, 객체, 텍스처
- **품질**: 고품질 이미지 사용

## 6. 문제 해결

### 일반적인 오류
1. **파일 경로 오류**: 이미지와 마스크 파일명이 정확히 매칭되는지 확인
2. **마스크 포맷 오류**: 마스크가 0과 255 값만 가지는지 확인
3. **크기 불일치**: 이미지와 마스크 크기가 동일한지 확인

### 데이터 검증 스크립트
```python
import os
import cv2
import glob

def validate_dataset(data_dir):
    """데이터셋 유효성 검사"""
    img_files = glob.glob(os.path.join(data_dir, '*.jpg'))
    
    for img_file in img_files:
        base_name = os.path.splitext(img_file)[0]
        mask_file = base_name + '_mask.png'
        
        if not os.path.exists(mask_file):
            print(f"❌ Missing mask: {mask_file}")
            continue
            
        img = cv2.imread(img_file)
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        
        if img.shape[:2] != mask.shape:
            print(f"❌ Size mismatch: {img_file}")
            continue
            
        unique_vals = set(mask.flatten())
        if not unique_vals.issubset({0, 255}):
            print(f"❌ Invalid mask values: {mask_file}")
            continue
            
        print(f"✅ Valid pair: {os.path.basename(img_file)}")

# 사용법
validate_dataset('/path/to/your/inference_data')
```

이 가이드를 따라 데이터를 준비하시면 LaMa 모델을 정상적으로 훈련하고 추론할 수 있습니다!