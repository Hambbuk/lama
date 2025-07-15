__version__ = "0.1.0"

# Re-export main CLI function so it can be invoked with `python -m cleanlama`.
from importlib import import_module as _imp


def _run():  # pragma: no cover
    """Entry point when running `python -m cleanlama`."""
    cli = _imp("cleanlama.cli")
    cli.main()