# Inpaint – Minimal LaMa Training & Inference

A lightweight wrapper around the original [LaMa](https://github.com/advimman/lama) implementation that keeps **only what you need**:

• 📚  single-file CLI for **training / inference / ONNX export**  
• 🖼️  no-frills dataset handling – just point to a folder  
• 📈  TensorBoard logs out of the box  

---

## 0. Try on Colab  
[Quick-start guide](examples/quickstart.md) | [![Run in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<org>/lama/blob/main/examples/quickstart.ipynb)

## 1. Installation
```bash
# one-liner (detects GPU & sets up env automatically)
bash scripts/setup_env.sh myenv   # or omit name → 'inpaint' env
```

## 2. Quick start

```bash
# Train (writes to ./experiments/<timestamp>)
python -m inpaint train --config configs/training/lama-fourier.yaml --gpus 1

# Watch training
tensorboard --logdir experiments

# Inference
python -m inpaint infer --checkpoint_dir ./experiments/20240101_123456 \
                       --input_dir ./demo_images --output_dir ./output

# ONNX export
python -m inpaint onnx --checkpoint_dir ./experiments/20240101_123456 \
                        --output model.onnx
```

Run any sub-command with `-h` for detailed options.

---

### Folder layout
```
inpaint/            # package: cli, tasks, models alias
configs/            # original YAML configs (feel free to prune further)
requirements.txt    # trimmed dependencies
README.md           # this file
```

Everything else from the original repo can be ignored or deleted if you wish – the wrappers dynamically import only the modules they need.

### FAQ
| Question | Answer |
|----------|--------|
| ModuleNotFoundError: models.ade20k | The stub lives in `inpaint/models/ade20k`. Ensure you didn't delete it and that you ran `bash scripts/setup_env.sh` which installs minimal stubs. |
| GPU driver newer than official wheel | Override with `CUDA_TAG=cu124 bash scripts/setup_env.sh` or fall back to CPU build automatically. |
