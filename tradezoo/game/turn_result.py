from dataclasses import dataclass
from typing import List

from tradezoo.agent import Action, DecisionBatch, Observation
from tradezoo.market import Trade
from .trader import Trader


@dataclass(frozen=True)
class TurnResult:
    trader: Trader
    observation: Observation
    decision_batch: DecisionBatch
    action: Action
    reward: float
    trades: List[Trade]
