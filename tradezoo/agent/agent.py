from dataclasses import dataclass
import torch

from .actor import Actor
from .critic import Critic


@dataclass(frozen=True)
class Agent:
    actor: Actor
    critic: Critic
