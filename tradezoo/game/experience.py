from dataclasses import dataclass
import numpy as np
from typing import List

from tradezoo.agent import Action, Agent, Observation


@dataclass(frozen=True)
class Experience:
    agent: Agent
    old_observation: Observation
    action: Action
    reward: float
    new_observation: Observation
