#!/usr/bin/env python3
"""Compatibility launcher for the installed ARC solver package."""

from pathlib import Path
import sys


# Avoid resolving this compatibility file as the ``arc_solver`` package when it
# is invoked directly from the repository root.
_REPOSITORY_ROOT = Path(__file__).resolve().parent
sys.path[:] = [
    entry for entry in sys.path if Path(entry or ".").resolve() != _REPOSITORY_ROOT
]

from arc_solver.cli import main


if __name__ == "__main__":
    main()
