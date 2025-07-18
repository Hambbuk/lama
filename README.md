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
The project uses **Hydra**. Configuration is split into two YAML files you normally edit:

1. **Global paths –** `configs/training/location/thumbnail.yaml`
   ```yaml
   # Absolute path to your dataset root (see folder tree below)
   data_root_dir: /absolute/path/to/thumbnail_dataset

   # Where checkpoints & logs are written (leave as-is or change)
   out_root_dir: ${env:PROJECT_ROOT}/experiments
   tb_dir:        ${env:PROJECT_ROOT}/tb_logs
   ```

2. **Data & augmentation –** `configs/training/data/hand_mask.yaml`
   ```yaml
   train:
     # RGB folders (one per date-range)
     indir:
       - ${location.data_root_dir}/field_thumbnail_dec12_dec21/train
       - ${location.data_root_dir}/field_thumbnail_sep01_sep03/train

     # Hand-mask folders — MUST match order above
     hand_mask_dir:
       - ${location.data_root_dir}/field_thumbnail_dec12_dec21/train_hand_mask
       - ${location.data_root_dir}/field_thumbnail_sep01_sep03/train_hand_mask

     kind: hand_mask_multi   # <- uses hand masks
     out_size: 256           # network input size in pixels
     mask_inflation: 21      # dilate hand mask (px) before merging
   ```

   **Common tweaks**
   * `batch_size`, `val_batch_size` – reduce if you hit OOM.
   * `transform_variant` – e.g. `distortions_light` for faster training.

**Calling scripts**
When you launch any script (`train.sh`, `inference.sh`, …) pass **config names without the `.yaml` extension**:
```bash
./scripts/train.sh -m lama-fourier -l thumbnail -d hand_mask   # OK
```

Hydra merges them in the order [model] → location → data.

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
    -c best.ckpt \                 # [optional] checkpoint file
    -o model.onnx \                # [optional] output path
    -s                              # run onnx-simplifier (default ON)
    -t ./demo                       # [optional] quick test with demo image & mask
```

---

## 7. Credits
* Original LaMa paper – Suvorov *et al.* 2021 ([arXiv:2109.07161](https://arxiv.org/abs/2109.07161))
* Code forked from the official repository: <https://github.com/advimman/lama>
* Hand masks generated with **MediaPipe Hands** (Apache 2.0).
