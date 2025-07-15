import importlib, sys

__version__ = "0.1.0"

# Alias existing saicinpainting package under inpaint.models to avoid massive moves
_saic = importlib.import_module("saicinpainting")

# Register alias
sys.modules[__name__ + ".models"] = _saic

# Ensure submodules resolve (e.g. inpaint.models.training)
for name, module in list(sys.modules.items()):
    if name.startswith("saicinpainting."):
        sys.modules[name.replace("saicinpainting", __name__ + ".models", 1)] = module

# Thin util re-export (optional)
from types import ModuleType
_utils = ModuleType("inpaint.utils")
setattr(_utils, "__doc__", "Utility helpers placeholder")
sys.modules["inpaint.utils"] = _utils