from dataclasses import dataclass
from functools import cached_property


@dataclass(frozen=True)
class Action:
    mid_price: float
    spread: float

    @cached_property
    def ask(self) -> float:
        return self.mid_price * (1 + self.spread)

    @cached_property
    def bid(self) -> float:
        return self.mid_price / (1 + self.spread)
