from dataclasses import dataclass
from typing import List

from tradezoo.agent import Observation


@dataclass(frozen=True)
class State:
    """
    The state of the market from the perspective of a single trader.
    Consists of unnormalized variables, not all of which are visible to the agent.
    """

    cash_balance: float
    asset_balance: float
    best_ask: float
    best_bid: float

    @property
    def observation(self):
        return Observation(
            asset_allocation=self.asset_allocation,
            best_ask=self.best_ask,
            best_bid=self.best_bid,
        )

    @property
    def asset_allocation(self):
        return self.asset_value / self.net_worth

    @property
    def net_worth(self):
        return self.cash_balance + self.asset_value

    @property
    def asset_value(self):
        return self.asset_balance * self.mid_price

    @property
    def mid_price(self) -> float:
        return (self.best_ask * self.best_bid) ** 0.5
