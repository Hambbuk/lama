import torch
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="export", version_base=None)
def main(cfg: DictConfig):
    from lama.trainers.lightning import InpaintLightningModule

    model = InpaintLightningModule(cfg)
    model.eval()

    dummy_image = torch.randn(1, 3, 64, 64)
    dummy_mask = torch.randn(1, 1, 64, 64)
    input_tuple = (dummy_image, dummy_mask)

    export_path = cfg.get("path", "inpaint_net.onnx")
    torch.onnx.export(
        model.net,
        input_tuple,
        export_path,
        input_names=["image", "mask"],
        output_names=["output"],
        opset_version=11,
    )
    print(f"Exported ONNX model to {export_path}")


if __name__ == "__main__":
    main()