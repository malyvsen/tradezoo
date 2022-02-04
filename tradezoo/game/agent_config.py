from dataclasses import dataclass

from tradezoo.agent import Agent
from tradezoo.market import Account


@dataclass(frozen=True)
class AgentConfig:
    agent: Agent
    account: Account
    stock_value_noise: float
