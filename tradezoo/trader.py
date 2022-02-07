from dataclasses import dataclass

from tradezoo.agent import Agent
from tradezoo.market import Account


@dataclass(frozen=True)
class Trader:
    """
    Everything that makes a market participant:
    An agent, its associated account, some configuration
    """

    agent: Agent
    account: Account
    stock_value_noise: float
    actor_learning_rate: float
    critic_learning_rate: float
