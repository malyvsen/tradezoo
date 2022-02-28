from dataclasses import dataclass
import numpy as np
from typing import List

from tradezoo.agent import Action, Observation
from tradezoo.game import TurnResult


@dataclass(frozen=True)
class Experience:
    old_observation: Observation
    action: Action
    reward: float
    new_observation: Observation

    @classmethod
    def from_turn_results(cls, old: TurnResult, new: TurnResult):
        assert old.trader == new.trader
        assert old.turn_number < new.turn_number
        return cls(
            old_observation=old.observation,
            action=old.action,
            reward=old.reward,
            new_observation=new.observation,
        )
