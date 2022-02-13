from dataclasses import dataclass

from tradezoo.agent import Agent
from tradezoo.market import Account
from .market_maker import MarketMaker


@dataclass
class Trader:
    agent: Agent
    account: Account
    market_maker: MarketMaker

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other
