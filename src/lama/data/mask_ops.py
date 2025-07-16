import torch
import numpy as np
from typing import Tuple


def random_hole_mask(
    size_hw: Tuple[int, int],
    hole_size_range: Tuple[int, int] = (32, 64),
) -> torch.Tensor:
    """Generate a random rectangular mask (0/1) tensor.

    Args:
        size_hw: (height, width) of the output mask.
        hole_size_range: min/max size of the hole edge in pixels.

    Returns:
        mask: (1, H, W) tensor where 1 indicates *hole* (to be inpainted).
    """
    h, w = size_hw
    mask = torch.zeros((1, h, w), dtype=torch.float32)

    hole_h = np.random.randint(*hole_size_range)
    hole_w = np.random.randint(*hole_size_range)

    top = np.random.randint(0, max(1, h - hole_h))
    left = np.random.randint(0, max(1, w - hole_w))

    mask[:, top : top + hole_h, left : left + hole_w] = 1.0
    return mask