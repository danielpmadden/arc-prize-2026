"""Run the production ARC solver with ``python -m renegade``."""

from pathlib import Path
import sys


# The root compatibility launcher is named ``arc_solver.py``. Exclude the
# checkout root so it cannot shadow the installed ``arc_solver`` package.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path[:] = [
    entry for entry in sys.path if Path(entry or ".").resolve() != _REPOSITORY_ROOT
]

from arc_solver.cli import main


if __name__ == "__main__":
    main()
