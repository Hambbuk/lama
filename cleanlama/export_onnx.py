import argparse
import logging
import os
from pathlib import Path

import torch
import yaml
from omegaconf import OmegaConf

from saicinpainting.training.trainers import load_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


class ONNXWrapper(torch.nn.Module):
    """Wrap original LaMa model to expose (image, mask)->inpainted signature."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor, mask: torch.Tensor):  # type: ignore[override]
        batch = {"image": image, "mask": mask}
        out = self.model(batch)
        return out["inpainted"]


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export LaMa checkpoint to ONNX (clean version)")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Directory produced by training that contains models/ and config.yaml")
    parser.add_argument("--checkpoint", type=str, default="best.ckpt", help="Checkpoint filename inside <checkpoint_dir>/models (default: best.ckpt)")
    parser.add_argument("--output", type=str, default="model.onnx", help="ONNX output path")
    parser.add_argument("--height", type=int, default=512, help="Input image height (default: 512)")
    parser.add_argument("--width", type=int, default=512, help="Input image width (default: 512)")
    parser.add_argument("--opset", type=int, default=16, help="ONNX opset version")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to run export on")
    return parser


def main() -> None:
    args = build_cli_parser().parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    cfg_path = os.path.join(args.checkpoint_dir, "config.yaml")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(cfg_path)
    with open(cfg_path, "r") as f:
        train_cfg = OmegaConf.create(yaml.safe_load(f))
    train_cfg.training_model.predict_only = True
    train_cfg.visualizer.kind = "noop"

    ckpt_path = os.path.join(args.checkpoint_dir, "models", args.checkpoint)
    LOGGER.info("Loading checkpoint %s", ckpt_path)
    model = load_checkpoint(train_cfg, ckpt_path, strict=False, map_location=device)
    model.eval()
    model.to(device)

    wrapper = ONNXWrapper(model).to(device)

    h, w = args.height, args.width
    dummy_img = torch.randn(1, 3, h, w, device=device)
    dummy_mask = torch.randn(1, 1, h, w, device=device)

    dynamic_axes = {
        "image": {2: "height", 3: "width"},
        "mask": {2: "height", 3: "width"},
        "inpainted": {2: "height", 3: "width"},
    }

    LOGGER.info("Exporting to ONNX %s", args.output)
    torch.onnx.export(
        wrapper,
        (dummy_img, dummy_mask),
        args.output,
        input_names=["image", "mask"],
        output_names=["inpainted"],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
    )
    LOGGER.info("ONNX export completed")


if __name__ == "__main__":
    main()