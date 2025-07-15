# Quick Start

```bash
# clone & cd into repo
bash scripts/setup_env.sh myenv  # GPU auto-detected, dependencies installed
source myenv/bin/activate        # or conda activate myenv

# 1. Train on a small config (CPU demo)
python -m inpaint train --config configs/train.yaml --max_steps 1 --gpus 0

# 2. Inference on demo folder
python -m inpaint infer \
  --checkpoint_dir runs/$(ls runs | head -1) \
  --input_dir demo_images --output_dir output --device cpu

# 3. Export to ONNX
python -m inpaint onnx \
  --checkpoint_dir runs/$(ls runs | head -1) --output lama.onnx
```