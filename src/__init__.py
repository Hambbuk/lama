import importlib, sys

__all__ = ["models"]
__version__ = "0.2.0"

# Expose saicinpainting under src.models for backward-compat
_saic_root = importlib.import_module("saicinpainting")

# Register direct package alias
sys.modules[__name__ + ".models"] = _saic_root

# Also map all its sub-modules (saicinpainting.* -> src.models.*)
for mod_name, mod in list(sys.modules.items()):
    if mod_name.startswith("saicinpainting"):
        new_name = mod_name.replace("saicinpainting", __name__ + ".models", 1)
        sys.modules[new_name] = mod