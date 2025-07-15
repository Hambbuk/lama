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

# Also expose as plain `models` for legacy code
import pkgutil as _pkgutil

sys.modules['models'] = _models

# Map submodules: models.xxx -> inpaint.models.xxx
for _mod in _pkgutil.walk_packages(_models.__path__, _models.__name__ + '.'):
    _alias_name = _mod.name.replace('inpaint.models', 'models', 1)
    if _alias_name not in sys.modules:
        try:
            sys.modules[_alias_name] = importlib.import_module(_mod.name)
        except ModuleNotFoundError:
            pass