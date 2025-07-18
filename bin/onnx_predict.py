#!/usr/bin/env python
"""Run inference with a LaMa ONNX model.

Example:
    python bin/onnx_predict.py \
        --model lama.onnx \
        --image ./examples/image.png \
        --mask  ./examples/image_mask.png \
        --output ./outputs/out.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

# -----------------------------------------------------------------------------

def preprocess_img(path: Path, size: int = 256) -> np.ndarray:
    """Load and preprocess an RGB image -> NCHW float32 0-1."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # CHW
    return img[np.newaxis]  # NCHW


def preprocess_mask(path: Path, size: int = 256, inflate: int | None = None) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)

    if inflate and inflate > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (inflate, inflate))
        mask = cv2.dilate(mask, kernel)

    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    mask = mask.astype(np.float32) / 255.0
    mask = mask[np.newaxis, np.newaxis, ...]  # NCHW
    return mask


def postprocess(pred: np.ndarray, out_path: Path):
    pred = np.clip(pred.squeeze(), 0, 1)
    pred = (pred * 255).astype(np.uint8)
    pred = np.transpose(pred, (1, 2, 0))  # HWC RGB
    pred_bgr = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), pred_bgr)


# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LaMa ONNX inference")
    parser.add_argument("--model", required=True, help="Path to .onnx model")
    parser.add_argument("--image", required=True, help="Path to RGB image")
    parser.add_argument("--mask", required=True, help="Path to grayscale mask")
    parser.add_argument("--output", required=True, help="Where to save result PNG")
    parser.add_argument("--inflate", type=int, default=0, help="Mask dilation radius (px)")
    args = parser.parse_args()

    img = preprocess_img(Path(args.image))
    mask = preprocess_mask(Path(args.mask), inflate=args.inflate)

    sess = ort.InferenceSession(args.model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    inputs = {sess.get_inputs()[0].name: img, sess.get_inputs()[1].name: mask}
    pred = sess.run(None, inputs)[0]  # assume first output is the image

    postprocess(pred, Path(args.output))
    print(f"Saved output → {args.output}")


if __name__ == "__main__":
    main()