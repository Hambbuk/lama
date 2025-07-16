# LaMa Inpainting

A clean and efficient implementation of LaMa (Large Mask Inpainting) for training, inference, and ONNX export.

## Features

- **Training**: Multi-GPU training with flexible configuration
- **Inference**: Fast inference with optional refinement
- **ONNX Export**: Model export for production deployment
- **Easy to Use**: Simple command-line interface

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install this package
pip install -e .
```

## Quick Start

### 1. Training

Train a LaMa model on your dataset:

```bash
python train.py \
    --config configs/training/big-lama.yaml \
    --data_path /path/to/your/dataset \
    --gpus 2 \
    --epochs 100 \
    --batch_size 8 \
    --output_dir ./outputs/training
```

#### Training Arguments:
- `--config`: Configuration file path
- `--gpus`: Number of GPUs to use
- `--data_path`: Path to training dataset
- `--output_dir`: Output directory for checkpoints and logs
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size per GPU
- `--learning_rate`: Learning rate
- `--resume`: Resume from checkpoint (optional)

### 2. Inference

Run inference on images with masks:

```bash
python inference.py \
    --checkpoint ./outputs/training/checkpoints/last.ckpt \
    --input /path/to/input/images \
    --output /path/to/output/results \
    --device cuda
```

#### Inference Arguments:
- `--checkpoint`: Path to model checkpoint
- `--input`: Input directory with images and masks
- `--output`: Output directory for results
- `--device`: Device to use (cuda/cpu)
- `--refine`: Enable refinement for better quality
- `--pad_modulo`: Pad input to modulo (default: 8)

### 3. ONNX Export

Export trained model to ONNX format:

```bash
python export_onnx.py \
    --checkpoint ./outputs/training/checkpoints/last.ckpt \
    --output ./models/lama_model.onnx \
    --input_size 512 512 \
    --simplify
```

#### Export Arguments:
- `--checkpoint`: Path to model checkpoint
- `--output`: Output ONNX file path
- `--input_size`: Input image size (height, width)
- `--batch_size`: Batch size for export
- `--dynamic_axes`: Enable dynamic input sizes
- `--simplify`: Simplify ONNX model

## Dataset Format

Your dataset should be organized as follows:

```
dataset/
├── images/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── masks/
    ├── img1_mask.png
    ├── img2_mask.png
    └── ...
```

- Images: RGB images in common formats (jpg, png, etc.)
- Masks: Binary masks where white (255) indicates areas to inpaint

## Configuration

The training configuration can be customized in `configs/training/` directory. Key settings include:

- **Model Architecture**: Generator and discriminator settings
- **Training Parameters**: Learning rates, loss weights, etc.
- **Data Processing**: Image augmentation and preprocessing
- **Hardware**: GPU settings and optimization

## Model Architecture

LaMa uses a FFC (Fast Fourier Convolution) based generator with:
- **Input**: 4 channels (RGB image + mask)
- **Output**: 3 channels (inpainted RGB image)
- **Architecture**: ResNet-like structure with FFC blocks
- **Training**: Adversarial training with perceptual and feature matching losses

## Performance Tips

### Training:
- Use multiple GPUs for faster training
- Adjust batch size based on GPU memory
- Enable mixed precision for better performance
- Use SSD storage for faster data loading

### Inference:
- Use GPU for faster inference
- Enable refinement for better quality (slower)
- Batch multiple images for efficiency
- Use appropriate padding for model requirements

### ONNX Export:
- Use dynamic axes for flexible input sizes
- Simplify model for better runtime performance
- Test exported model with sample inputs
- Consider quantization for deployment

## Troubleshooting

### Common Issues:

1. **Out of Memory**: Reduce batch size or use gradient accumulation
2. **Slow Training**: Check data loading bottlenecks, use SSD storage
3. **Poor Results**: Adjust loss weights, check dataset quality
4. **ONNX Export Fails**: Check model compatibility, reduce complexity

### Performance Optimization:

```bash
# Enable optimizations
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1

# For better memory usage
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

## Directory Structure

```
lama-inpainting/
├── train.py              # Training script
├── inference.py          # Inference script
├── export_onnx.py        # ONNX export script
├── requirements.txt      # Dependencies
├── configs/              # Configuration files
├── saicinpainting/       # Core modules
│   ├── training/         # Training modules
│   ├── evaluation/       # Evaluation modules
│   └── utils.py          # Utilities
└── README.md            # This file
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{suvorov2021resolution,
  title={Resolution-robust Large Mask Inpainting with Fourier Convolutions},
  author={Suvorov, Roman and Logacheva, Elizaveta and Mashikhin, Anton and Remizova, Anastasia and Ashukha, Arseniy and Silvestrov, Aleksei and Kong, Naejin and Goka, Harshith and Park, Kiwoong and Lempitsky, Victor},
  journal={arXiv preprint arXiv:2109.07161},
  year={2021}
}
```
