from dataclasses import dataclass
import numpy as np
from typing import List

from .geometric_brownian_motion import GeometricBrownianMotion
from .turn_result import TurnResult
from tradezoo.market import BuyOrder, Market, SellOrder
from tradezoo.agent import Agent, Observation


@dataclass
class Game:
    market: Market
    stock_value: GeometricBrownianMotion
    agents: List[Agent]
    whose_turn: int

    @classmethod
    def new(
        cls, market: Market, stock_value: GeometricBrownianMotion, agents: List[Agent]
    ) -> "Game":
        return cls(market=market, stock_value=stock_value, agents=agents, whose_turn=0)

    def turn_(self) -> TurnResult:
        current_agent = self.agents[self.whose_turn]
        for own_order in self.market.orders_by(current_agent.account):
            self.market.cancel_(own_order)

        observation = Observation.from_situation(
            market=self.market,
            account=current_agent.account,
            true_stock_value=self.stock_value.value,
            noise=np.random.normal(loc=0, scale=current_agent.stock_value_noise),
        )
        (action,) = current_agent.decide(observation.batch).sample()
        self.market.submit_(
            BuyOrder(submitted_by=current_agent.account, price=action.bid, volume=1)
        )
        self.market.submit_(
            SellOrder(submitted_by=current_agent.account, price=action.ask, volume=1)
        )

        self.stock_value.step_()
        self.whose_turn = (self.whose_turn + 1) % len(self.agents)
        return TurnResult(
            agent=current_agent,
            observation=observation,
            action=action,
            reward=current_agent.account.net_worth(self.stock_value.value),
        )
