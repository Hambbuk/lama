import sys
from pathlib import Path

# Ensure `src` directory is on PYTHONPATH so `import lama` works without installation
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))