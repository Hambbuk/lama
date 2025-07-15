import sys
import argparse
import importlib
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)

SUB_MODULES = {
    "train": "src.train",
    "infer": "src.infer",
    "onnx": "src.export_onnx",
}

def main():
    parser = argparse.ArgumentParser(
        prog="cleanlama",  # preserve command name for backward compatibility
        description="Unified CLI: train / infer / onnx export",
    )
    parser.add_argument("command", choices=SUB_MODULES.keys(), help="Subcommand: {train, infer, onnx}")
    args, remainder = parser.parse_known_args()

    module_name = SUB_MODULES[args.command]
    LOGGER.debug("Dispatching to %s", module_name)
    module = importlib.import_module(module_name)

    sys.argv = [args.command] + remainder
    if hasattr(module, "main"):
        module.main()  # type: ignore
    else:
        LOGGER.error("%s has no main()", module_name)
        sys.exit(1)

if __name__ == "__main__":
    main()