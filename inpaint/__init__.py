import importlib, sys

__version__ = "0.1.0"

# Alias existing saicinpainting package under inpaint.models to avoid massive moves
_models = importlib.import_module("inpaint.models")

# Expose it as top-level alias too (optional)
sys.modules[__name__ + ".models"] = _models

# Also make old 'saicinpainting' refer to same module for backward-compat
sys.modules["saicinpainting"] = _models
_models.__name__ = "inpaint.models"  # ensure correct pkg name

# Thin util re-export (optional)
from types import ModuleType
_utils = ModuleType("inpaint.utils")
setattr(_utils, "__doc__", "Utility helpers placeholder")
sys.modules["inpaint.utils"] = _utils