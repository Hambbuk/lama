# LaMa (Slimmed)

This repository is a minimal, production-grade fork of the original **LaMa** image-inpainting project.  
Everything you need for **training**, **inference** and **ONNX export** is kept ‑ all the rest is gone.

---

## Quick start

```bash
# clone
$ git clone https://github.com/<your-org>/lama.git && cd lama

# 1. create conda env (≈ 2 mins)
$ conda env create -f environment.yml
$ conda activate lama

# 2. add the correct PyTorch build for your GPU (≈ 30 s)
$ source scripts/post_install.sh

# 3. train – change anything at runtime via Hydra!
$ python bin/train.py trainer.kwargs.gpus=4

# 4. inference
$ python bin/predict.py \
        model.path=/path/to/checkpoint_dir \
        indir=/path/to/images_and_masks \
        outdir=/tmp/out

# 5. ONNX export (for deployment)
$ python scripts/export_onnx.py \
        model.path=/path/to/checkpoint_dir \
        save_path=model.onnx \
        image_size=512
```

---

## Folder layout

```
├── bin/               # core entrypoints kept from the upstream repo
│   ├── train.py       # train with PyTorch-Lightning (+DDP)
│   └── predict.py     # batched inference
├── scripts/
│   ├── export_onnx.py # export trained checkpoint to ONNX
│   └── post_install.sh# install the right PyTorch build after env creation
├── saicinpainting/    # model implementation (unchanged)
├── configs/           # Hydra configs
├── requirements.txt   # python deps (except torch*)
└── environment.yml    # clean conda spec (python + pip only)
```

Everything else (Dockerfiles, notebooks, legacy dataset scripts …) was removed to keep the code base lean.

---

## Multi-GPU & DDP

LaMa already relies on **PyTorch-Lightning**, so scaling to multiple GPUs is as easy as:

```bash
python bin/train.py trainer.kwargs.gpus=<N>  # e.g. gpus=8
```

All other trainer flags (precision, gradient clipping …) can be appended the same way.

---

## Environment details

1. `environment.yml` installs only **Python 3.10** and **pip**.
2. `scripts/post_install.sh` detects the driver CUDA version and pulls the matching wheel from the official PyTorch index.
3. The remaining Python requirements are installed via `pip -r requirements.txt` automatically by `conda`.

> You can of course adapt this flow to poetry / pip-tools / docker.

---

## License

The original authors released LaMa under the **Apache 2.0** license – this fork keeps the same terms.
