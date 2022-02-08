from dataclasses import dataclass

from tradezoo.agent import Action, Observation
from tradezoo.trader import Trader


@dataclass(frozen=True)
class TurnResult:
    trader: Trader
    observation: Observation
    action: Action
    reward: float
