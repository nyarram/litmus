"""Make the repo root importable so ``import examples.*`` works under plain ``pytest``.

(``python -m pytest`` adds CWD to ``sys.path`` implicitly; the ``pytest``
console script used in CI does not.)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
