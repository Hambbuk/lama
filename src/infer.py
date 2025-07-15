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

from src.models.evaluation.utils import move_to_device  # type: ignore
from src.models.training.data.datasets import make_default_val_dataset  # type: ignore
from src.models.training.trainers import load_checkpoint  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inference with a LaMa checkpoint (clean)")
    parser.add_argument("--checkpoint_dir", required=True, help="Directory with models/ and config.yaml")
    parser.add_argument("--checkpoint", default="best.ckpt", help="Checkpoint filename inside models/")
    parser.add_argument("--input_dir", required=True, help="Folder with images + masks")
    parser.add_argument("--output_dir", required=True, help="Where to write results")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--out_ext", default=".png")
    return parser


def main():
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
    LOGGER.info("Loading %s", ckpt_path)
    model = load_checkpoint(train_cfg, ckpt_path, strict=False, map_location=device)
    model.freeze()
    model.to(device)

    dataset = make_default_val_dataset(args.input_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    LOGGER.info("Dataset: %d images", len(dataset))

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
        res = np.clip(res * 255, 0, 255).astype(np.uint8)
        res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
        fname = os.path.splitext(os.path.basename(dataset.image_filenames[idx]))[0] + args.out_ext
        cv2.imwrite(os.path.join(args.output_dir, fname), res)


if __name__ == "__main__":
    main()