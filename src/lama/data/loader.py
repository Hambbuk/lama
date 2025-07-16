import os
from pathlib import Path
from typing import Callable, Optional, List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from .mask_ops import random_hole_mask


class FolderPairDataset(Dataset):
    """A dataset that loads (image, mask) pairs from two separate folders.

    Expected directory layout::
        root/
            images/
                xxx.png
            masks/
                xxx.png
    where the same filename exists under both *images* and *masks* folder.
    """

    def __init__(
        self,
        root: str | Path,
        image_folder: str = "images",
        mask_folder: str = "masks",
        image_transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        root = Path(root)
        self.images_root = root / image_folder
        self.masks_root = root / mask_folder

        image_files = sorted(list(self.images_root.glob("*")))
        if not image_files:
            raise RuntimeError(f"No images found in {self.images_root}")

        self._paths: List[Tuple[Path, Path]] = []
        for img_path in image_files:
            mask_path = self.masks_root / img_path.name
            if mask_path.exists():
                self._paths.append((img_path, mask_path))

        if not self._paths:
            raise RuntimeError("No (image, mask) pairs with matching filenames found.")

        self.image_transform = image_transform or T.ToTensor()
        self.mask_transform = mask_transform or T.ToTensor()

    def __len__(self) -> int:  # noqa: D401
        return len(self._paths)

    def __getitem__(self, idx: int):  # noqa: D401
        img_path, mask_path = self._paths[idx]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        return {
            "image": self.image_transform(image),
            "mask": self.mask_transform(mask),
        }


class InpaintingDataset(Dataset):
    """Dataset that returns an image and a *generated* binary mask suitable for inpainting."""

    def __init__(
        self,
        root: str | Path,
        transform: Optional[Callable] = None,
        mask_size: tuple[int, int] | None = None,
        hole_size_range: tuple[int, int] = (32, 64),
    ) -> None:
        super().__init__()
        self.folder_dataset = FolderPairDataset(root=root)
        self.transform = transform or T.Compose([
            T.Resize(mask_size) if mask_size else T.Lambda(lambda x: x),
            T.ToTensor(),
        ])
        self.hole_size_range = hole_size_range

    def __len__(self):  # noqa: D401
        return len(self.folder_dataset)

    def __getitem__(self, idx):  # noqa: D401
        sample = self.folder_dataset[idx]
        image = sample["image"]
        _, h, w = image.shape
        mask = random_hole_mask((h, w), self.hole_size_range)
        return {
            "image": image,
            "mask": mask,
        }