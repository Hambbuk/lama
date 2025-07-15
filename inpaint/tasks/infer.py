import argparse
import logging
import os

import cv2
import numpy as np
import torch
import tqdm
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

from ..models.evaluation.utils import move_to_device  # type: ignore
from ..models.training.data.datasets import make_default_val_dataset  # type: ignore
from ..models.training.trainers import load_checkpoint  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Inference (inpaint)")
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--checkpoint", default="best.ckpt")
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--out_ext", default=".png")
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

    ckpt_path = os.path.join(args.checkpoint_dir, "models", args.checkpoint)
    model = load_checkpoint(cfg, ckpt_path, strict=False, map_location=device)
    model.freeze().to(device)

    dataset = make_default_val_dataset(args.input_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    for idx in tqdm.trange(len(dataset), desc="infer"):
        batch = default_collate([dataset[idx]])
        batch = move_to_device(batch, device)
        batch["mask"] = (batch["mask"] > 0).float()
        with torch.no_grad():
            batch = model(batch)
            res = batch["inpainted"][0].permute(1, 2, 0).cpu().numpy()
            if (unpad := batch.get("unpad_to_size")) is not None:
                h, w = unpad
                res = res[:h, :w]
        res = cv2.cvtColor(np.clip(res * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        fname = os.path.splitext(os.path.basename(dataset.image_filenames[idx]))[0] + args.out_ext
        cv2.imwrite(os.path.join(args.output_dir, fname), res)


if __name__ == "__main__":
    main()