from dataclasses import dataclass

from tradezoo.agent import Agent
from tradezoo.market import Account


@dataclass
class Trader:
    """
    Everything that makes a market participant:
    An agent, its associated account, some configuration
    """

    agent: Agent
    account: Account
    stock_value_noise: float

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other
