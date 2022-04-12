from dataclasses import dataclass
from typing import List

from tradezoo.game import TurnResult


@dataclass(frozen=True)
class TrainResult:
    loss: float
