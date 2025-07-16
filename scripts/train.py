import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    from lama.trainers.lightning import InpaintLightningModule
    model = InpaintLightningModule(cfg)
    trainer = pl.Trainer(max_epochs=cfg.get("max_epochs", 1), accelerator="cpu", logger=False)
    trainer.fit(model)


if __name__ == "__main__":
    main()