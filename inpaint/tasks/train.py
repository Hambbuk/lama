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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Train LaMa (inpaint)")
    p.add_argument("--config", required=True, help="YAML config path")
    p.add_argument("--workdir", default="runs", help="Output root directory")
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--precision", type=int, choices=[16, 32], default=32)
    p.add_argument("--max_steps", type=int)
    p.add_argument("--resume", type=str)
    return p


def cli() -> None:
    main()


def main():
    args = build_parser().parse_args()

    exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    workdir = os.path.abspath(os.path.join(args.workdir, exp_name))
    os.makedirs(workdir, exist_ok=True)

    cfg = OmegaConf.load(args.config)
    OmegaConf.save(cfg, os.path.join(workdir, "config.yaml"))

    model = make_training_model(cfg)

    callbacks = [
        ModelCheckpoint(dirpath=os.path.join(workdir, "models"), save_top_k=3, monitor="val_loss", mode="min"),
        LearningRateMonitor(logging_interval="step"),
    ]
    tb_logger = TensorBoardLogger(save_dir=workdir, name="tb")

    trainer = pl.Trainer(
        gpus=args.gpus if args.gpus > 0 else None,
        precision=args.precision,
        default_root_dir=workdir,
        callbacks=callbacks,
        logger=tb_logger,
        max_steps=args.max_steps,
        resume_from_checkpoint=args.resume,
    )
    trainer.fit(model)


if __name__ == "__main__":
    main()