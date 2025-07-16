import torch
from lama.models.inpaint_net import InpaintNet


def test_forward_pass():
    model = InpaintNet()
    image = torch.rand(2, 3, 64, 64)
    mask = (torch.rand(2, 1, 64, 64) > 0.5).float()
    out = model(image, mask)
    assert out.shape == image.shape