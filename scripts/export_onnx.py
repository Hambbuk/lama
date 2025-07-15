#!/usr/bin/env python3
"""Export a trained LaMa checkpoint to ONNX.

Example:
    python scripts/export_onnx.py \
        model.path=/path/to/big-lama \
        model.checkpoint=last.ckpt \
        image_size=512 \
        save_path=big-lama.onnx

The script re-uses Hydra configs from configs/prediction/default.yaml so all usual
prediction parameters remain available. Only *save_path* (output file name) and
*image_size* (square side length for dummy input) must be provided or will fall
back to sensible defaults.
"""
from __future__ import annotations

import os
import sys
import logging
from pathlib import Path

import hydra
import torch
import yaml
from omegaconf import OmegaConf

from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.utils import register_debug_signal_handlers

LOGGER = logging.getLogger(__name__)


class ONNXWrapper(torch.nn.Module):
    """Thin wrapper matching the usual inference signature for ONNX export."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self._model = model

    def forward(self, image: torch.Tensor, mask: torch.Tensor):  # noqa: D401
        batch = {"image": image, "mask": mask}
        out = self._model(batch)
        return out["inpainted"]


@hydra.main(config_path="../configs/prediction", config_name="default.yaml")
def main(cfg: OmegaConf):  # noqa: C901, D401
    if sys.platform != "win32":
        register_debug_signal_handlers()

    # Resolve parameters -----------------------------------------------------
    ckpt_dir = Path(cfg.model.path).expanduser().resolve()
    ckpt_path = ckpt_dir / "models" / cfg.model.checkpoint
    save_path = Path(cfg.get("save_path", ckpt_dir / "model.onnx"))
    image_size = int(cfg.get("image_size", 256))

    LOGGER.info("Loading checkpoint from %s", ckpt_path)

    # Re-create train config --------------------------------------------------
    with open(ckpt_dir / "config.yaml", "r") as f:
        train_cfg = OmegaConf.create(yaml.safe_load(f))
    train_cfg.training_model.predict_only = True
    train_cfg.visualizer.kind = "noop"

    model = load_checkpoint(train_cfg, ckpt_path, strict=False, map_location="cpu")
    model.eval()

    wrapper = ONNXWrapper(model)

    # Dummy inputs -----------------------------------------------------------
    dummy_image = torch.randn(1, 3, image_size, image_size, requires_grad=False)
    dummy_mask = torch.randn(1, 1, image_size, image_size, requires_grad=False)

    # Choose export device ---------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapper = wrapper.to(device)
    dummy_image, dummy_mask = dummy_image.to(device), dummy_mask.to(device)

    # Export -----------------------------------------------------------------
    LOGGER.info("Exporting to ONNX at %s", save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        (dummy_image, dummy_mask),
        str(save_path),
        input_names=["image", "mask"],
        output_names=["inpainted"],
        dynamic_axes={
            "image": {2: "height", 3: "width"},
            "mask": {2: "height", 3: "width"},
            "inpainted": {2: "height", 3: "width"},
        },
        opset_version=16,
    )

    LOGGER.info("✅ Successfully saved ONNX model to %s", save_path)


if __name__ == "__main__":
    main()