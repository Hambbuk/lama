from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torchvision.transforms as T

from lama.models.inpaint_net import InpaintNet
from lama.data.loader import InpaintingDataset


class InpaintLightningModule(pl.LightningModule):
    """A minimal Lightning module wrapping `InpaintNet`.

    The loss is simply \( \ell_1 \) between output and ground-truth image outside the mask area –
    good enough for CI purposes.
    """

    def __init__(self, cfg: Any | None = None):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])
        self.cfg = cfg or {}
        model_kwargs = self.cfg.get("model", {})
        self.net = InpaintNet(**model_kwargs)
        self.lr = self.cfg.get("lr", 1e-3)

        # simple transform for dummy dataset
        self.default_transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
        ])

    # ---------------------------------------------------------------------
    # Forward & loss
    # ---------------------------------------------------------------------
    def forward(self, image: torch.Tensor, mask: torch.Tensor):  # type: ignore[override]
        return self.net(image, mask)

    def training_step(self, batch, batch_idx):  # noqa: D401
        image, mask = batch["image"], batch["mask"]
        output = self(image, mask)
        loss = torch.nn.functional.l1_loss(output * mask, image * mask)
        self.log("train/loss", loss)
        return loss

    # ---------------------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------------------
    def configure_optimizers(self):  # noqa: D401
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    # ---------------------------------------------------------------------
    # Dataloaders (dummy synthetic dataset)
    # ---------------------------------------------------------------------
    def _make_dataset(self, train: bool):
        # Use synthetic random data – only for quick CI.
        class _RandomImages(torch.utils.data.Dataset):
            def __len__(self):
                return 64  # small dataset

            def __getitem__(self, idx):  # noqa: D401
                img = torch.rand(3, 64, 64)
                mask = (torch.rand(1, 64, 64) > 0.8).float()
                return {"image": img, "mask": mask}

        return _RandomImages()

    def train_dataloader(self):  # noqa: D401
        return DataLoader(self._make_dataset(True), batch_size=8, shuffle=True, num_workers=0)