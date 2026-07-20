from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

Grid = tuple[tuple[int, ...], ...]
PredictFn = Callable[[Grid], Optional[Grid]]


@dataclass(frozen=True)
class Rule:
    name: str
    priority: int
    predict: PredictFn
