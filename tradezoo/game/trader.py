from dataclasses import dataclass

from tradezoo.agent import Agent
from tradezoo.market import Account, Market
from .client import Client


@dataclass
class Trader:
    agent: Agent
    account: Account
    client: Client

    def utility(self, market: Market):
        mid_price = (
            market.best_ask(self.account) * market.best_bid(self.account)
        ) ** 0.5
        return self.account.cash_balance + self.account.asset_balance * mid_price

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other
