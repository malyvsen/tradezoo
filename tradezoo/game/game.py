from dataclasses import dataclass
import numpy as np
from typing import List

from .geometric_brownian_motion import GeometricBrownianMotion
from .agent_config import AgentConfig
from .turn_result import TurnResult
from tradezoo.market import BuyOrder, Market, SellOrder
from tradezoo.agent import Observation


@dataclass
class Game:
    market: Market
    stock_value: GeometricBrownianMotion
    agent_configs: List[AgentConfig]
    whose_turn: int

    def turn_(self) -> TurnResult:
        current_config = self.agent_configs[self.whose_turn]
        for own_order in self.market.orders_by(current_config.account):
            self.market.cancel_(own_order)

        observation = Observation.from_situation(
            market=self.market,
            account=current_config.account,
            true_stock_value=self.stock_value.value,
            noise=np.random.normal(loc=0, scale=current_config.stock_value_noise),
        )
        (action,) = current_config.agent.decide(observation.batch).sample()
        self.market.submit_(
            BuyOrder(submitted_by=current_config.account, price=action.bid, volume=1)
        )
        self.market.submit_(
            SellOrder(submitted_by=current_config.account, price=action.ask, volume=1)
        )

        self.stock_value.step_()
        self.whose_turn = (self.whose_turn + 1) % len(self.agent_configs)
        return TurnResult(
            agent=current_config.agent,
            observation=observation,
            action=action,
            reward=current_config.account.net_worth(self.stock_value.value),
        )
