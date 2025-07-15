import argparse
import logging
import os
from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import OmegaConf

from src.models.training.trainers import make_training_model

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train LaMa model (clean)")
    parser.add_argument("--config", required=True, help="Training YAML config path")
    parser.add_argument("--workdir", default="experiments", help="Root directory for experiments")
    parser.add_argument("--gpus", type=int, default=1, help="#GPUs (0 = CPU)")
    parser.add_argument("--precision", type=int, default=32, choices=[16, 32])
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume")
    return parser


def main():
    args = build_cli_parser().parse_args()

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    workdir = os.path.abspath(os.path.join(args.workdir, run_name))
    os.makedirs(workdir, exist_ok=True)

    config = OmegaConf.load(args.config)
    OmegaConf.save(config, os.path.join(workdir, "config.yaml"))

    model = make_training_model(config)

    checkpoint_dir = os.path.join(workdir, "models")
    callbacks = [
        ModelCheckpoint(dirpath=checkpoint_dir, save_last=True, save_top_k=3, monitor="val_loss", mode="min"),
        LearningRateMonitor(logging_interval="step"),
    ]
    logger = TensorBoardLogger(save_dir=workdir, name="tb_logs")

    trainer_kwargs = dict(
        gpus=args.gpus if args.gpus > 0 else None,
        precision=args.precision,
        default_root_dir=workdir,
        callbacks=callbacks,
        logger=logger,
    )
    if args.max_steps:
        trainer_kwargs["max_steps"] = args.max_steps
    if args.resume:
        trainer_kwargs["resume_from_checkpoint"] = args.resume

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model)


if __name__ == "__main__":
    main()