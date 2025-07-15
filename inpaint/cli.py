import click

@click.group()
def main():
    """Inpainting toolkit: train / infer / onnx"""

@main.command(context_settings={"ignore_unknown_options": True})
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def train(args):
    """Run training (delegates to inpaint.tasks.train)"""
    import sys, inpaint.tasks.train as mod
    sys.argv = ["train", *args]
    mod.main()

@main.command(context_settings={"ignore_unknown_options": True})
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def infer(args):
    """Run inference (delegates to inpaint.tasks.infer)"""
    import sys, inpaint.tasks.infer as mod
    sys.argv = ["infer", *args]
    mod.main()

@main.command(context_settings={"ignore_unknown_options": True})
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def onnx(args):
    """Export checkpoint to ONNX (delegates to inpaint.tasks.export_onnx)"""
    import sys, inpaint.tasks.export_onnx as mod
    sys.argv = ["onnx", *args]
    mod.main()

if __name__ == "__main__":
    main()