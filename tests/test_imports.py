import importlib, pkgutil, pytest

roots = [
    'inpaint',
    'models',
    'saicinpainting'
]

@pytest.mark.parametrize('module_name', [m for root in roots for m in ([root] + [s.name for s in pkgutil.walk_packages(importlib.import_module(root).__path__, root + '.')])])
def test_import(module_name):
    importlib.import_module(module_name)