# CleanLaMa – Minimal LaMa Training & Inference

A lightweight wrapper around the original [LaMa](https://github.com/advimman/lama) implementation that keeps **only what you need**:

• 📚  single-file CLI for **training / inference / ONNX export**  
• 🖼️  no-frills dataset handling – just point to a folder  
• 📈  TensorBoard logs out of the box  

---

## 1. Installation
```bash
python -m venv venv && source venv/bin/activate  # optional but recommended
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118  # pick CUDA / CPU build you need
pip install -r requirements.txt
```

## 2. Quick start

```bash
# Train (writes to ./experiments/<timestamp>)
python -m src train --config configs/training/lama-fourier.yaml --gpus 1

# Watch training
tensorboard --logdir experiments

# Inference
python -m src infer --checkpoint_dir ./experiments/20240101_123456 \
                       --input_dir ./demo_images --output_dir ./output

# ONNX export
python -m src onnx --checkpoint_dir ./experiments/20240101_123456 \
                        --output model.onnx
```

Run any sub-command with `-h` for detailed options.

---

### Folder layout
```
cleanlama/          # minimal CLI implementation (train / infer / onnx)
configs/            # original YAML configs (feel free to prune further)
requirements.txt    # trimmed dependencies
README.md           # this file
```

Everything else from the original repo can be ignored or deleted if you wish – the wrappers dynamically import only the modules they need.
