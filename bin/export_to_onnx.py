#!/usr/bin/env python3
"""Export a trained LaMa run (Lightning checkpoint) to ONNX.

Example
-------
python bin/export_to_onnx.py \
    --model-dir ./experiments/2025-07-17_train_lama-fourier_ \
    --checkpoint best.ckpt \
    --output lama.onnx \
    --test --test-data ./examples
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence, Tuple, Union

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from yaml import safe_load

from saicinpainting.training.trainers.default import DefaultInpaintingTrainingModule

# -----------------------------------------------------------------------------
# Helper wrappers
# -----------------------------------------------------------------------------

class ExportLamaModule(torch.nn.Module):
    """Wraps a loaded LaMa Lightning module for ONNX export."""

    def __init__(self, lama_module: DefaultInpaintingTrainingModule):
        super().__init__()
        self.model = lama_module

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Predict inpainted image (uint8 0-255)."""
        masked_img = image * (1 - mask)
        if self.model.concat_mask:
            masked_img = torch.cat([masked_img, mask], dim=1)
        predicted = self.model.generator(masked_img)
        inpainted = mask * predicted + (1 - mask) * image
        return torch.clamp(inpainted * 255, min=0, max=255)


# -----------------------------------------------------------------------------
# Exporter
# -----------------------------------------------------------------------------

class LamaOnnxExporter:
    """Loads a trained run (config + checkpoint) and exports to ONNX."""

    def __init__(self, model_dir: Path, device: str = "cpu"):
        self.model_dir = model_dir
        self.device = torch.device(device)
        self.config = None
        self.lama_module: DefaultInpaintingTrainingModule | None = None
        self.export_model: ExportLamaModule | None = None

    # ------------------------------------------------------------------
    def load(self, checkpoint_name: str = "best.ckpt"):
        cfg_path = self.model_dir / "config.yaml"
        ckpt_path = self.model_dir / "models" / checkpoint_name
        if not cfg_path.exists() or not ckpt_path.exists():
            raise FileNotFoundError("Could not find config.yaml or checkpoint under", self.model_dir)

        self.config = OmegaConf.create(safe_load(cfg_path.read_text()))
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

        # Disable perceptual loss if it was enabled during training
        if "resnet_pl" in self.config.losses:
            self.config.losses.resnet_pl.weight = 0

        kwargs = dict(self.config.training_model)
        kwargs.pop("kind", None)
        kwargs["use_ddp"] = True
        self.config.generator.resnet_conv_kwargs.use_jit = True

        m = DefaultInpaintingTrainingModule(self.config, **kwargs)
        m.load_state_dict(state["state_dict"], strict=False)
        m.on_load_checkpoint(state)
        m.eval().freeze()
        self.lama_module = m

        self.export_model = ExportLamaModule(m).eval().to(self.device)

    # ------------------------------------------------------------------
    def export(self, out_path: Path, img_size: int = 256, opset: int = 13):
        assert self.export_model, "Model not loaded"
        img = torch.rand(1, 3, img_size, img_size, device=self.device)
        msk = torch.rand(1, 1, img_size, img_size, device=self.device)

        torch.onnx.export(
            self.export_model,
            (img, msk),
            str(out_path),
            input_names=["image", "mask"],
            output_names=["output"],
            export_params=True,
            do_constant_folding=True,
            opset_version=opset,
            dynamic_axes={"image": {0: "batch"}, "mask": {0: "batch"}, "output": {0: "batch"}},
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _inflate(mask: np.ndarray, k: int) -> np.ndarray:
        if k <= 0:
            return mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        return cv2.dilate(mask, kernel)

    def test(self, onnx_path: Path, demo_dir: Path, inflate: int = 0):
        img_p = demo_dir / "image.png"
        msk_p = demo_dir / "image_mask.png"
        if not (img_p.exists() and msk_p.exists()):
            print("[WARN] demo image/mask not found, skipping test")
            return

        img = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (256, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[None, ...]

        msk = cv2.imread(str(msk_p), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        msk = self._inflate(msk, inflate)
        msk = cv2.resize(msk, (256, 256))
        msk = msk[None, None, ...]

        sess = ort.InferenceSession(str(onnx_path))
        out = sess.run(None, {"image": img, "mask": msk})[0][0]

        out = np.clip(out, 0, 255).astype(np.uint8)
        out = np.transpose(out, (1, 2, 0))
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(onnx_path.with_suffix("_demo.png")), out)
        print("[INFO] Demo prediction saved →", onnx_path.with_suffix("_demo.png"))


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Export LaMa run to ONNX")
    p.add_argument("--model-dir", "-m", required=True, help="LaMa experiment directory")
    p.add_argument("--checkpoint", "-c", default="best.ckpt", help="ckpt filename inside models/")
    p.add_argument("--output", "-o", default="lama.onnx", help="Output ONNX file")
    p.add_argument("--test", action="store_true", help="Run quick ONNX sanity-test after export")
    p.add_argument("--test-data", default="./examples", help="Folder with image.png & image_mask.png")
    p.add_argument("--inflate", type=int, default=0, help="Mask dilation pixels for test run")
    return p.parse_args()


def main():
    args = parse_args()
    exporter = LamaOnnxExporter(Path(args.model_dir))
    exporter.load(args.checkpoint)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    exporter.export(out_path)
    print(f"[INFO] ONNX model saved → {out_path}")

    if args.test:
        exporter.test(out_path, Path(args.test_data), inflate=args.inflate)


if __name__ == "__main__":
    main()