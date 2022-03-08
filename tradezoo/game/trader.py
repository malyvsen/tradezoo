from dataclasses import dataclass
import math

from tradezoo.agent import Agent, Observation
from tradezoo.market import Account
from .client import Client


@dataclass
class Trader:
    agent: Agent
    account: Account
    client: Client

    def utility(self, observation: Observation) -> float:
        mid_price = (observation.best_ask * observation.best_bid) ** 0.5
        net_worth = observation.cash_balance + observation.asset_balance * mid_price
        return math.log(1 + net_worth)

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other
