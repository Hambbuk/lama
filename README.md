# LaMa: Large Mask Inpainting

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training
```bash
python tool/train.py model.path=config/training/big-lama.yaml
```

### Inference
```bash
python tool/predict.py model.path=path/to/checkpoint.ckpt indir=input_images outdir=output_images
```

### ONNX Export
```bash
python tool/to_jit.py config/training/big-lama.yaml path/to/checkpoint.ckpt output.onnx
```

## Configuration

- `config/training/` - Training configurations
- `config/prediction/` - Prediction configurations  
- `config/eval*.yaml` - Evaluation configurations

## Project Structure

```
├── config/          # Configuration files
├── src/            # Source code
├── tool/           # Main scripts (train, predict, export)
├── util/           # Utility scripts
├── requirements.txt
└── README.md
```
