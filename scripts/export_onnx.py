"""Export a trained LaMa model to ONNX.

Example
-------
$ python scripts/export_onnx.py \
    --model-dir ./big-lama \
    --checkpoint best.ckpt \
    --output model.onnx

This script loads the training config & checkpoint via saicinpainting utilities,
creates the inference network and exports it to ONNX with dynamic axes so that
it can run on variable-size images.
"""
import argparse
import os
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf

from saicinpainting.training.trainers import make_training_model


def load_model(model_dir: str, checkpoint: str):
    cfg_path = os.path.join(model_dir, 'config.yaml')
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Cannot find config.yaml in {model_dir}")
    cfg = OmegaConf.load(cfg_path)
    ckpt_path = os.path.join(model_dir, checkpoint)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model = make_training_model(cfg)
    state = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state['state_dict'], strict=False)
    model.eval()
    return model


def export(model, output_path: str, opset: int = 17):
    dummy = torch.randn(1, 4, 256, 256)  # 3 channels image + 1 channel mask
    dynamic_axes = {
        'input': {2: 'height', 3: 'width'},
        'output': {2: 'height', 3: 'width'},
    }
    torch.onnx.export(
        model,
        dummy,
        output_path,
        export_params=True,
        opset_version=opset,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
    )
    print(f"ONNX model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export LaMa model to ONNX")
    parser.add_argument('--model-dir', required=True, help='Directory with a trained model (contains config.yaml)')
    parser.add_argument('--checkpoint', default='best.ckpt', help='Checkpoint file name inside model-dir')
    parser.add_argument('--output', default='lama.onnx', help='Output ONNX file path')
    args = parser.parse_args()

    model = load_model(args.model_dir, args.checkpoint)
    export(model, args.output)


if __name__ == '__main__':
    pl.seed_everything(42, workers=True)
    main()