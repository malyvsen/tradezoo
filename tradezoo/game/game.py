from dataclasses import dataclass
from typing import List

from .geometric_brownian_motion import GeometricBrownianMotion
from .ownership import Ownership
from .turn_result import TurnResult
from tradezoo.market import BuyOrder, Market, SellOrder
from tradezoo.agent import State


@dataclass
class Game:
    market: Market
    stock_value: GeometricBrownianMotion
    ownerships: List[Ownership]
    whose_turn: int

    def turn_(self) -> TurnResult:
        current_agent = self.ownerships[self.whose_turn].agent
        current_account = self.ownerships[self.whose_turn].account

        for own_order in self.market.orders_by(current_account):
            self.market.cancel_(own_order)
        state = State.from_situation(
            market=self.market,
            account=current_account,
            stock_value=self.stock_value.value,
        )
        (action,) = current_agent.decide(state.batch).sample()
        self.market.submit_(
            BuyOrder(submitted_by=current_account, price=action.bid, volume=1)
        )
        self.market.submit_(
            SellOrder(submitted_by=current_account, price=action.ask, volume=1)
        )

        self.stock_value.step_()
        self.whose_turn = (self.whose_turn + 1) % len(self.ownerships)
        return TurnResult(
            agent=current_agent,
            state=state,
            action=action,
            reward=current_account.net_worth(self.stock_value.value),
        )
