from dataclasses import dataclass
import random
from typing import List

from .experience import Experience
from tradezoo.game import TurnResult


@dataclass
class ReplayBuffer:
    capacity: int
    last_turn_result: TurnResult
    experiences: List[Experience]

    @classmethod
    def empty(cls, capacity: int):
        return cls(capacity=capacity, last_turn_result=None, experiences=[])

    def sample(self, num_experiences: int) -> List[Experience]:
        return random.choices(self.experiences, k=num_experiences)

    def register_turn_(self, turn_result: TurnResult):
        if self.last_turn_result is None:
            return
        self.add_experience_(
            Experience.from_turn_results(old=self.last_turn_result, new=turn_result)
        )
        self.last_turn_result = turn_result

    def add_experience_(self, experience: Experience):
        self.experiences.append(experience)
        if len(self.experiences) > self.capacity:
            self.experiences.pop(random.randrange(0, len(self.experiences)))
