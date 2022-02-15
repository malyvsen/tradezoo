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
    whose_turn: int

    @classmethod
    def new(cls, market: Market, traders: List[Trader]) -> "Game":
        return cls(market=market, traders=traders, whose_turn=0)

    def turn_(self) -> TurnResult:
        trader = self.traders[self.whose_turn]
        for own_order in self.market.orders_by(trader.account):
            self.market.cancel_(own_order)

        for market_maker_order in trader.market_maker.orders():
            self.market.submit_(market_maker_order)
        observation = Observation.from_situation(
            market=self.market,
            account=trader.account,
        )
        (action,) = trader.agent.decide(observation.batch).sample()
        buy_trades = self.market.submit_(
            BuyOrder(submitted_by=trader.account, price=action.bid, volume=1)
        )
        sell_trades = self.market.submit_(
            SellOrder(submitted_by=trader.account, price=action.ask, volume=1)
        )
        for market_maker_order in self.market.orders_by(trader.market_maker.account):
            self.market.cancel_(market_maker_order)

        self.whose_turn = (self.whose_turn + 1) % len(self.traders)
        return TurnResult(
            trader=trader,
            observation=observation,
            action=action,
            reward=trader.account.net_worth(
                stock_value=(observation.best_bid * observation.best_ask) ** 0.5
            ),
            trades=buy_trades + sell_trades,
        )
