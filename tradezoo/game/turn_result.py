from dataclasses import dataclass
from typing import List

from tradezoo.agent import Decision, ObservationSeries
from tradezoo.market import Market, Order, Trade
from .state import State
from .trader import Trader


@dataclass(frozen=True)
class BaseTurnResult:
    trader: Trader
    observations: ObservationSeries

    def next_(self, market: Market, time_step: int):
        for client_order in market.orders_by(self.trader.client.account):
            market.cancel_(client_order)
        for own_order in market.orders_by(self.trader.account):
            market.cancel_(own_order)
        for client_order in self.trader.client.orders(time_step):
            market.submit_(client_order)

        state = self.trader.state(market=market)
        observations = self.observations.with_new(
            state.observation, horizon=self.trader.agent.horizon
        )
        decision = self.trader.agent.decide(observations)
        order = self.trader.order(state=state, decision=decision)
        trades = market.submit_(order)
        return TurnResult(
            trader=self.trader,
            observations=observations,
            time_step=time_step,
            state=state,
            decision=decision,
            order=order,
            trades=trades,
        )


@dataclass(frozen=True)
class InitialTurnResult(BaseTurnResult):
    @classmethod
    def empty(cls, trader: Trader):
        return cls(trader=trader, observations=ObservationSeries(observations=[]))


@dataclass(frozen=True)
class TurnResult(BaseTurnResult):
    time_step: int
    state: State
    decision: Decision
    order: Order
    trades: List[Trade]
