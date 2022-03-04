from dataclasses import dataclass
from typing import List

from tradezoo.agent import Observation
from tradezoo.market import BuyOrder, Market, SellOrder
from .trader import Trader
from .turn_result import TurnResult


@dataclass
class Game:
    market: Market
    traders: List[Trader]
    turn_number: int

    @classmethod
    def new(cls, market: Market, traders: List[Trader]) -> "Game":
        return cls(market=market, traders=traders, turn_number=0)

    def turn_(self) -> TurnResult:
        trader = self.traders[self.turn_number % len(self.traders)]
        for client_order in self.market.orders_by(trader.client.account):
            self.market.cancel_(client_order)
        for own_order in self.market.orders_by(trader.account):
            self.market.cancel_(own_order)

        for client_order in trader.client.orders(self.turn_number):
            self.market.submit_(client_order)
        old_utility = trader.utility(self.market)
        observation = Observation.from_situation(
            market=self.market,
            account=trader.account,
        )
        decision_batch = trader.agent.decide(observation.batch)
        (action,) = decision_batch.sample()
        buy_trades = self.market.submit_(
            BuyOrder.public(submitted_by=trader.account, price=action.bid, volume=1)
        )
        sell_trades = self.market.submit_(
            SellOrder.public(submitted_by=trader.account, price=action.ask, volume=1)
        )
        new_utility = trader.utility(self.market)

        result = TurnResult(
            turn_number=self.turn_number,
            trader=trader,
            observation=observation,
            decision_batch=decision_batch,
            action=action,
            reward=new_utility - old_utility,
            trades=buy_trades + sell_trades,
        )
        self.turn_number += 1
        return result
