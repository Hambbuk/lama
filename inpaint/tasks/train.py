import argparse
import logging
import os
from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import OmegaConf

from ..models.training.trainers import make_training_model  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser("Train an inpainting model (LaMa)")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--workdir", default="runs", help="Directory to store experiments")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--precision", type=int, choices=[16, 32], default=32)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(args.workdir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    cfg = OmegaConf.load(args.config)
    OmegaConf.save(cfg, os.path.join(exp_dir, "config.yaml"))

    model = make_training_model(cfg)

    callbacks = [
        ModelCheckpoint(dirpath=os.path.join(exp_dir, "models"), save_top_k=3, monitor="val_loss", mode="min"),
        LearningRateMonitor(logging_interval="step"),
    ]
    tb_logger = TensorBoardLogger(save_dir=exp_dir, name="tb")

    trainer = pl.Trainer(
        gpus=args.gpus if args.gpus > 0 else None,
        precision=args.precision,
        default_root_dir=exp_dir,
        callbacks=callbacks,
        logger=tb_logger,
        max_steps=args.max_steps,
        resume_from_checkpoint=args.resume,
    )
    trainer.fit(model)

if __name__ == "__main__":
    main()