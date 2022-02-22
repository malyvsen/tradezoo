from dataclasses import dataclass

from tradezoo.agent import Agent
from tradezoo.market import Account
from .client import Client


@dataclass
class Trader:
    agent: Agent
    account: Account
    client: Client

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other
