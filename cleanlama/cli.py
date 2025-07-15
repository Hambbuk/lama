import sys
import argparse
import importlib
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


SUB_MODULES = {
    "train": "cleanlama.train",
    "infer": "cleanlama.inference",
    "onnx": "cleanlama.export_onnx",
}


def main():
    parser = argparse.ArgumentParser(
        prog="cleanlama",
        description="Unified CLI for training, inference and ONNX export of LaMa models (clean edition)",
    )
    parser.add_argument(
        "command",
        choices=SUB_MODULES.keys(),
        help="Subcommand to run: {train, infer, onnx}",
    )
    # everything after the first positional arg is passed to the submodule unchanged
    args, remainder = parser.parse_known_args()

    module_name = SUB_MODULES[args.command]
    LOGGER.debug("Dispatching to %s with argv=%s", module_name, remainder)
    module = importlib.import_module(module_name)

    # dispatch by emulating command-line of the submodule
    sys.argv = [args.command] + remainder
    if hasattr(module, "main"):
        module.main()  # type: ignore
    else:
        LOGGER.error("Selected module %s has no main() entrypoint", module_name)
        sys.exit(1)


if __name__ == "__main__":
    main()