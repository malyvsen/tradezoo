from dataclasses import dataclass

from tradezoo.agent import Agent
from tradezoo.market import Account


@dataclass(frozen=True)
class Ownership:
    agent: Agent
    account: Account
