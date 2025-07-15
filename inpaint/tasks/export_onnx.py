import argparse
import logging
import os

import torch
import yaml
from omegaconf import OmegaConf

from ..models.training.trainers import load_checkpoint  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor, mask: torch.Tensor):  # type: ignore[override]
        return self.model({"image": image, "mask": mask})["inpainted"]


def build_parser():
    p = argparse.ArgumentParser("Export ONNX")
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--checkpoint", default="best.ckpt")
    p.add_argument("--output", default="inpaint.onnx")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--opset", type=int, default=16)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return p


def cli():
    main()


def main():
    args = build_parser().parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    cfg_path = os.path.join(args.checkpoint_dir, "config.yaml")
    with open(cfg_path, "r") as f:
        cfg = OmegaConf.create(yaml.safe_load(f))
    cfg.training_model.predict_only = True
    cfg.visualizer.kind = "noop"

    ckpt = os.path.join(args.checkpoint_dir, "models", args.checkpoint)
    model = load_checkpoint(cfg, ckpt, strict=False, map_location=device)
    model.eval().to(device)

    wrapper = Wrapper(model).to(device)
    dummy_img = torch.randn(1, 3, args.height, args.width, device=device)
    dummy_mask = torch.randn(1, 1, args.height, args.width, device=device)

    torch.onnx.export(
        wrapper,
        (dummy_img, dummy_mask),
        args.output,
        input_names=["image", "mask"],
        output_names=["inpainted"],
        dynamic_axes={"image": {2: "h", 3: "w"}, "mask": {2: "h", 3: "w"}, "inpainted": {2: "h", 3: "w"}},
        opset_version=args.opset,
    )
    LOGGER.info("Saved %s", args.output)


if __name__ == "__main__":
    main()