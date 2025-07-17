import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    from saicinpainting.training.trainers import make_training_model

    model = make_training_model(cfg)
    # Use all available GPUs; fall back to CPU automatically handled by Lightning
    strategy = "ddp" if pl.utilities.device_parser.num_cuda_devices() > 1 else "auto"
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=-1,
        strategy=strategy,
        max_epochs=cfg.trainer.max_epochs,
        default_root_dir=cfg.out_dir,
    )
    trainer.fit(model)


if __name__ == "__main__":
    main()