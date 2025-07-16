
# Import aliases for legacy `saicinpainting` paths used in old notebooks.
import importlib
import sys

_MAP = {
    "saicinpainting.training.modules.ffc": "lama.models.blocks.ffc",
    "saicinpainting.training.modules.spatial_transform": "lama.models.blocks.spatial_transform",
    "saicinpainting.training.losses.fft": "lama.losses.fft",
    "saicinpainting.training.losses.feature_matching": "lama.losses.adversarial",
    "saicinpainting.training.losses.perceptual": "lama.losses.extra.perceptual",
    "saicinpainting.training.datasets.places": "lama.data.extra.places",
}

for _old, _new in _MAP.items():
    try:
        if _old not in sys.modules:
            sys.modules[_old] = importlib.import_module(_new)
    except ModuleNotFoundError:
        # Skip alias if new module relies on heavy optional deps not installed.
        pass