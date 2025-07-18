# Thumbnail Inpainting – Quick Start

> Lightweight guide to train, test, and export the LaMa-based thumbnail inpainting model with hand-mask support.

---

## 1  Install Requirements
```bash
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install --upgrade pip
pip install -r requirements.txt                   # core deps

# (GPU) pick the wheel matching your CUDA version, e.g. CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 2  Prepare Dataset
```
${data_root_dir}/field_thumbnail_<date_range>/
├─ train/               *.jpg
├─ train_hand_mask/     *.png   # same base-name as JPG
├─ val/                 image + *_mask.png pairs
└─ visual_test/         (optional) image + *_mask.png
```

---

## 3  Edit Configs
1. `configs/training/location/thumbnail.yaml`
   ```yaml
   data_root_dir: /abs/path/to/thumbnail_dataset
   ```
2. `configs/training/data/hand_mask.yaml`
   * `train.indir` & `train.hand_mask_dir` – date-range folders you own
   * `batch_size`, `mask_inflation` – tweak for your GPU

> 모든 설정은 Hydra 구문(`${location.data_root_dir}` 등)를 사용하므로 **파일 확장자는 빼고** `-m/-l/-d` 옵션으로 지정합니다.

---

## 4  Train
```bash
chmod +x scripts/*.sh               # 최초 1회 권한 부여
./scripts/train.sh                  # 기본값: lama-fourier / thumbnail / hand_mask
# 커스텀 예시
# ./scripts/train.sh -m lama-fourier -l thumbnail -d hand_mask
```
체크포인트와 로그는 `experiments/` 하위에 자동 저장됩니다.

---

## 5  Inference
```bash
./scripts/inference.sh \
    -m ./experiments/<exp_dir> \
    -i ./demo \
    -o ./outputs
```
`-i/-o` 생략 시 `./demo`, `./outputs` 가 기본값입니다.

---

## 6  Export to ONNX
```bash
./scripts/export_to_onnx.sh \
    -m ./experiments/<exp_dir> \   # 모델 디렉터리
    -c best.ckpt \                 # or last.ckpt
    -o model.onnx \
    -s                              # onnx-simplifier 적용 (선택)
```

---

## 7  Credits
* LaMa – Suvorov et al., 2021 (MIT License)
* Code base forked from <https://github.com/advimman/lama>
* Hand masks generated via MediaPipe Hands (Apache 2.0)
