# LaMa: Resolution-robust Large Mask Inpainting with Fourier Convolutions

Official PyTorch implementation of the paper:

**Resolution-robust Large Mask Inpainting with Fourier Convolutions**

[arXiv](https://arxiv.org/abs/2109.07161) | [Project Page](https://saic-mdal.github.io/lama-project/) | [Colab](https://colab.research.google.com/github/saic-mdal/lama/blob/master/colab/LaMa_inpainting.ipynb)

## Features

- 🚀 **High-quality inpainting** for large masks
- 🎯 **Resolution-robust** - works on various image sizes  
- ⚡ **Fast inference** with GPU acceleration
- 🔧 **Easy to use** - simple installation and usage
- 📦 **ONNX export** support for deployment

## Installation

### Quick Install (Recommended)

The setup script automatically detects your CUDA version and installs appropriate PyTorch:

```bash
git clone https://github.com/your-org/lama.git
cd lama
pip install -e .
```

For full installation with all features:
```bash
pip install -e ".[full]"
```

### Manual Install

If you prefer manual installation:

```bash
# For CUDA 11.6
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html

# For CPU only
pip install torch==1.12.1+cpu torchvision==0.13.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Install other dependencies
pip install -r requirements.txt
```

## Quick Start

### Download Pretrained Model

Download pretrained Big-LaMa model:

```bash
mkdir -p pretrained
wget https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip -P pretrained/
unzip pretrained/big-lama.zip -d pretrained/
```

### Inference

#### Single Image
```bash
python scripts/predict.py \
    --model pretrained/big-lama \
    --image examples/image.jpg \
    --mask examples/mask.jpg \
    --output output.jpg
```

#### Batch Processing
```bash
python scripts/predict.py \
    --model pretrained/big-lama \
    --input-dir ./images \
    --output-dir ./results
```

### Training

```bash
# Single GPU
python scripts/train.py --config configs/default.yaml

# Multi-GPU (e.g., 4 GPUs)
python scripts/train.py --config configs/default.yaml --gpus 4

# Custom settings
python scripts/train.py \
    --config configs/default.yaml \
    --gpus 2 \
    --batch-size 16 \
    --max-epochs 200 \
    --exp-name my_experiment
```

### ONNX Export

```bash
python scripts/export_onnx.py \
    --model pretrained/big-lama \
    --output lama.onnx \
    --simplify \
    --verify
```

## Project Structure

```
lama/
├── lama/                  # Core library
│   ├── training/         # Training modules
│   ├── evaluation/       # Evaluation tools
│   └── utils/           # Utilities
├── scripts/             # Executable scripts
│   ├── train.py        # Training script
│   ├── predict.py      # Inference script
│   └── export_onnx.py  # ONNX export
├── configs/            # Configuration files
├── experiments/        # Training outputs (auto-created)
└── examples/           # Example images
```

## Configuration

Edit `configs/default.yaml` to customize:

- **Model architecture**: Change `generator` parameters
- **Training settings**: Adjust batch size, learning rate, epochs
- **Loss weights**: Modify loss function weights
- **Data paths**: Set your dataset locations

## Dataset Preparation

Organize your dataset as follows:

```
data/
├── train/
│   ├── images/    # Training images
│   └── masks/     # Training masks
└── val/
    ├── images/    # Validation images  
    └── masks/     # Validation masks
```

Masks should be:
- Binary images (0 for known, 255 for missing regions)
- Same filename as corresponding image

## Advanced Usage

### Multi-GPU Training

```bash
# Use all available GPUs
python scripts/train.py --config configs/default.yaml --gpus -1

# Use specific GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train.py --config configs/default.yaml --gpus 4
```

### Mixed Precision Training

```bash
python scripts/train.py --config configs/default.yaml --fp16
```

### Resume Training

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --resume-from experiments/my_experiment/checkpoints/last.ckpt
```

### Custom Masks

For inference with custom masks:
- White pixels (255) = regions to inpaint
- Black pixels (0) = known regions

## Performance Tips

1. **Batch Size**: Adjust based on GPU memory
   - V100 (16GB): batch_size=10-15
   - RTX 3090 (24GB): batch_size=15-20
   
2. **Image Size**: Model works best at 256x256 or 512x512

3. **Inference Speed**: Use `--refine` for better quality (slower)

## Citation

```bibtex
@inproceedings{suvorov2021resolution,
  title={Resolution-robust Large Mask Inpainting with Fourier Convolutions},
  author={Suvorov, Roman and Logacheva, Elizaveta and Mashikhin, Anton and 
          Remizova, Anastasia and Ashukha, Arsenii and Silvestrov, Aleksei and
          Kong, Naejin and Goka, Harshith and Park, Kiwoong and Lempitsky, Victor},
  booktitle={WACV},
  year={2022}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Original implementation by [SAIC-Moscow](https://github.com/saic-mdal/lama)
- Pretrained models from [HuggingFace](https://huggingface.co/smartywu/big-lama)