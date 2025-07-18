# Thumbnail Inpainting – Quick Start

This is a **minimal guide** for training, testing, and exporting the LaMa-based thumbnail inpainting model (with optional hand-mask constraints).

---

## 1. Install Requirements (Python 3.10 + pip)
```bash
# 1) ensure Python 3.10 is active (e.g. via pyenv or system installation)
python --version  # → 3.10.x

# 2) create an optional virtual environment
python -m venv venv && source venv/bin/activate

# 3) install all Python packages
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4) GPU users – swap in a CUDA-enabled PyTorch wheel if needed:
#    pip uninstall -y torch torchvision
#    # CUDA 11.8 example
#    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Cloud GPUs (Paperspace, Kaggle, HF Spaces)**
1. The base image must have NVIDIA drivers compatible with CUDA 11.8 (or 12.1 if you install that wheel).
2. Run the same `pip install -r requirements.txt` inside your workspace/ container.
---

## 2. Prepare the Dataset
```
${data_root_dir}/field_thumbnail_<date_range>/
├─ train/               *.jpg
├─ train_hand_mask/     *.png   # single-channel hand masks (same basename)
├─ val/                 image + *_mask.png pairs
└─ visual_test/         (optional) image + *_mask.png pairs
```

---

## 3. Edit Configs
1. `configs/training/location/thumbnail.yaml`
   ```yaml
   data_root_dir: /absolute/path/to/thumbnail_dataset
   ```
2. `configs/training/data/hand_mask.yaml`
   * Update `train.indir` and `train.hand_mask_dir` to match your date ranges.
   * Tune `batch_size`, `mask_inflation`, etc. if needed.

> All files rely on Hydra interpolation (e.g. `${location.data_root_dir}`). When calling the shell scripts pass **names without the `.yaml` extension** via `-m`, `-l`, `-d`.

---

## 4. Train
```bash
# one-time permission fix
chmod +x scripts/*.sh

# default: model=lama-fourier, location=thumbnail, data=hand_mask
./scripts/train.sh

# example override
# ./scripts/train.sh -m lama-fourier -l thumbnail -d hand_mask
```
Checkpoints and TensorBoard logs are written under `experiments/`.

---

## 5. Inference
```bash
./scripts/inference.sh \
    -m ./experiments/<exp_dir> \
    -i ./demo \
    -o ./outputs
```
If `-i` / `-o` are omitted the script falls back to `./demo` and `./outputs`.

---

## 6. Export to ONNX
```bash
./scripts/export_to_onnx.sh \
    -m ./experiments/<exp_dir> \   # experiment directory
    -c best.ckpt \                 # or last.ckpt
    -o model.onnx \                # output file name
    -s                              # (optional) run onnx-simplifier
```

---

## 7. Credits
* Original LaMa paper – Suvorov *et al.* 2021 ([arXiv:2109.07161](https://arxiv.org/abs/2109.07161))
* Code forked from the official repository: <https://github.com/advimman/lama>
* Hand masks generated with **MediaPipe Hands** (Apache 2.0).
