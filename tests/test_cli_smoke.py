import subprocess, sys, tempfile, pathlib

CONFIG = pathlib.Path('configs/train.yaml')

def _run(module_args):
    return subprocess.run([sys.executable, '-m', *module_args], capture_output=True, text=True, check=True)

def test_cli_help():
    _run(['inpaint', '--help'])

def test_train_one_step(tmp_path):
    if not CONFIG.exists():
        return
    _run(['inpaint', 'train', '--config', CONFIG.as_posix(), '--max_steps', '1', '--gpus', '0', '--workdir', tmp_path.as_posix()])