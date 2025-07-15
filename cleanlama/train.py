import argparse
import logging
import os
from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import OmegaConf

from saicinpainting.training.trainers import make_training_model

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def build_cli_parser() -> argparse.ArgumentParser:
    """Return argument parser for the training CLI."""
    parser = argparse.ArgumentParser(description="Train LaMa inpainting model (clean version)")
    parser.add_argument("--config", type=str, required=True, help="Path to training YAML config file")
    parser.add_argument("--workdir", type=str, default="experiments", help="Directory where experiment artefacts will be stored")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use (0 for CPU)")
    parser.add_argument("--precision", type=int, default=32, choices=[16, 32], help="Floating-point precision (16 or 32)")
    parser.add_argument("--max_steps", type=int, default=None, help="Max training steps (overrides config if set)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from (optional)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


def main() -> None:
    parser = build_cli_parser()
    args = parser.parse_args()

    # Prepare working directory
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    workdir = os.path.abspath(os.path.join(args.workdir, run_name))
    os.makedirs(workdir, exist_ok=True)

    # Load config
    config = OmegaConf.load(args.config)
    OmegaConf.save(config, os.path.join(workdir, "config.yaml"))

    # Instantiate training model from the original repo helpers
    training_model = make_training_model(config)

    # Callbacks & loggers
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
    if args.max_steps is not None:
        trainer_kwargs["max_steps"] = args.max_steps
    if args.resume is not None:
        trainer_kwargs["resume_from_checkpoint"] = args.resume

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(training_model)


if __name__ == "__main__":
    main()