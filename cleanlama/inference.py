import argparse
import os
import logging

import cv2
import numpy as np
import torch
import tqdm
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.training.data.datasets import make_default_val_dataset
from saicinpainting.training.trainers import load_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run inference with a LaMa checkpoint (clean version)")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Directory produced by training that contains models/ and config.yaml")
    parser.add_argument("--checkpoint", type=str, default="best.ckpt", help="Checkpoint file name inside <checkpoint_dir>/models (default: best.ckpt)")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with input images & masks (name pattern *_mask*.png)")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save completed images")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="Device to run inference on")
    parser.add_argument("--out_ext", type=str, default=".png", help="Output image extension")
    return parser


def main() -> None:
    parser = build_cli_parser()
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load train config to re-create model
    train_cfg_path = os.path.join(args.checkpoint_dir, "config.yaml")
    if not os.path.isfile(train_cfg_path):
        raise RuntimeError(f"config.yaml not found at {train_cfg_path}")
    with open(train_cfg_path, "r") as f:
        train_cfg = OmegaConf.create(yaml.safe_load(f))
    train_cfg.training_model.predict_only = True  # tell trainer we only need forward
    train_cfg.visualizer.kind = "noop"

    ckpt_path = os.path.join(args.checkpoint_dir, "models", args.checkpoint)
    LOGGER.info("Loading checkpoint %s", ckpt_path)
    model = load_checkpoint(train_cfg, ckpt_path, strict=False, map_location=device)
    model.freeze()
    model.to(device)

    dataset = make_default_val_dataset(args.input_dir)
    LOGGER.info("Dataset size: %d", len(dataset))

    os.makedirs(args.output_dir, exist_ok=True)
    for idx in tqdm.trange(len(dataset), desc="processing"):
        batch = default_collate([dataset[idx]])
        batch = move_to_device(batch, device)
        batch["mask"] = (batch["mask"] > 0).float()
        with torch.no_grad():
            batch = model(batch)
            res = batch["inpainted"][0].permute(1, 2, 0).detach().cpu().numpy()
            unpad = batch.get("unpad_to_size", None)
            if unpad is not None:
                h, w = unpad
                res = res[:h, :w]
        res = np.clip(res * 255, 0, 255).astype(np.uint8)
        res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
        fname = dataset.image_filenames[idx]
        fname = os.path.splitext(os.path.basename(fname))[0] + args.out_ext
        cv2.imwrite(os.path.join(args.output_dir, fname), res)


if __name__ == "__main__":
    main()