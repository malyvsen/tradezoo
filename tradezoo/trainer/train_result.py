from dataclasses import dataclass
from typing import List

from tradezoo.game import TurnResult


@dataclass(frozen=True)
class TrainResult:
    actor_loss: float
    critic_loss: float


@dataclass(frozen=True)
class OnlineTrainResult:
    turn_result: TurnResult
    train_results: List[TrainResult]
