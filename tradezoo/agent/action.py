import math
from dataclasses import dataclass
from functools import cached_property


@dataclass(frozen=True)
class Action:
    log_mid_price: float
    log_spread: float

    @cached_property
    def mid_price(self):
        return math.exp(self.log_mid_price)

    @cached_property
    def spread(self):
        return math.exp(self.log_spread)

    @cached_property
    def ask(self) -> float:
        return self.mid_price * (1 + self.spread)

    @cached_property
    def bid(self) -> float:
        return self.mid_price / (1 + self.spread)
