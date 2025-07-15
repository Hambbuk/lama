import click
import importlib
import sys

@click.group()
def main():
    """Unified CLI for the inpaint toolkit"""

@main.command(context_settings={"ignore_unknown_options": True})
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def train(args):
    """Run training."""
    module = importlib.import_module("inpaint.tasks.train")
    sys.argv = ["train", *args]
    module.main()

@main.command(context_settings={"ignore_unknown_options": True})
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def infer(args):
    """Run inference."""
    module = importlib.import_module("inpaint.tasks.infer")
    sys.argv = ["infer", *args]
    module.main()

@main.command(context_settings={"ignore_unknown_options": True})
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def onnx(args):
    """Export checkpoint to ONNX."""
    module = importlib.import_module("inpaint.tasks.export_onnx")
    sys.argv = ["onnx", *args]
    module.main()

if __name__ == "__main__":
    main()