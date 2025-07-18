# Thumbnail Inpainting – Quick Start

This is a **minimal guide** for training, testing, and exporting the LaMa-based thumbnail inpainting model (with optional hand-mask constraints).

---

## 1. Install Requirements (Python 3.10, Conda + `conda_env.yml`)
```bash
# 1) create the environment from the YAML (fixed to Python 3.10)
conda env create -f conda_env.yml   # env name defaults to "lama"

# 2) activate
conda activate lama

# 3) (optional) upgrade pip & install extras
python -m pip install --upgrade pip
# pip install -r requirements.txt    # only if you modified the YAML

# 4) GPU users – switch the PyTorch wheel if necessary:
#    pip uninstall -y torch torchvision
#    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118   # CUDA 11.8 example
```

**Running on cloud GPUs (e.g. Paperspace, Kaggle, HuggingFace Spaces)**
1. Make sure the base image has CUDA drivers that match the wheel in `conda_env.yml` (CUDA 11.8 by default).
2. In Spaces / Docker, call the same two commands:
   ```bash
   conda env create -f conda_env.yml && conda activate lama
   ```
3. If conda is unavailable, fall back to:
   ```bash
   pip install -r requirements.txt
   ```

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
