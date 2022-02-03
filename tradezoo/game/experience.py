from dataclasses import dataclass
import numpy as np
from typing import List

from tradezoo.agent import Action, Agent, State


@dataclass(frozen=True)
class Experience:
    agent: Agent
    old_state: State
    action: Action
    reward: float
    new_state: State
