from dataclasses import dataclass
import numpy as np
from typing import List

from .geometric_brownian_motion import GeometricBrownianMotion
from .turn_result import TurnResult
from tradezoo.trader import Trader
from tradezoo.market import BuyOrder, Market, SellOrder
from tradezoo.agent import Observation


@dataclass
class Game:
    market: Market
    stock_value: GeometricBrownianMotion
    traders: List[Trader]
    whose_turn: int

    @classmethod
    def new(
        cls, market: Market, stock_value: GeometricBrownianMotion, traders: List[Trader]
    ) -> "Game":
        return cls(
            market=market, stock_value=stock_value, traders=traders, whose_turn=0
        )

    def turn_(self) -> TurnResult:
        current_trader = self.traders[self.whose_turn]
        for own_order in self.market.orders_by(current_trader.account):
            self.market.cancel_(own_order)

        observation = Observation.from_situation(
            market=self.market,
            account=current_trader.account,
            true_stock_value=self.stock_value.value,
            noise=np.random.normal(loc=0, scale=current_trader.stock_value_noise),
        )
        (action,) = current_trader.agent.decide(observation.batch).sample()
        self.market.submit_(
            BuyOrder(submitted_by=current_trader.account, price=action.bid, volume=1)
        )
        self.market.submit_(
            SellOrder(submitted_by=current_trader.account, price=action.ask, volume=1)
        )

        self.stock_value.step_()
        self.whose_turn = (self.whose_turn + 1) % len(self.traders)
        return TurnResult(
            agent=current_trader.agent,
            observation=observation,
            action=action,
            reward=current_trader.account.net_worth(self.stock_value.value),
        )
