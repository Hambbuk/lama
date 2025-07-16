import hydra
from omegaconf import DictConfig
import torch
import torchvision.utils as vutils


@hydra.main(config_path="../configs", config_name="infer", version_base=None)
def main(cfg: DictConfig):
    from lama.trainers.lightning import InpaintLightningModule

    # Load checkpoint if provided else random weights
    model = InpaintLightningModule.load_from_checkpoint(cfg.ckpt) if cfg.get("ckpt") else InpaintLightningModule(cfg)
    model.eval()

    # Dummy inference with random data (for example only)
    image = torch.rand(1, 3, 64, 64)
    mask = (torch.rand(1, 1, 64, 64) > 0.8).float()
    with torch.no_grad():
        output = model(image=image, mask=mask)

    vutils.save_image(output, "prediction.png")
    print("Saved prediction to prediction.png")


if __name__ == "__main__":
    main()