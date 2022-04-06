from dataclasses import dataclass
from typing import List, Dict

from tradezoo.market import Market
from .state import State, StateSeries
from .trader import Trader
from .turn_result import InitialTurnResult, TurnResult


@dataclass
class Game:
    """Responsible for scheduling which trader gets to trade when."""

    market: Market
    turn_results: List[TurnResult]
    time_step: int

    @classmethod
    def new(cls, market: Market, traders: List[Trader]) -> "Game":
        return cls(
            market=market,
            turn_results=[InitialTurnResult.empty(trader=trader) for trader in traders],
            time_step=0,
        )

    def turn_(self) -> TurnResult:
        index = self.time_step % len(self.turn_results)
        old_result = self.turn_results[index]
        new_result = old_result.next(market=self.market, time_step=self.time_step)
        self.turn_results[index] = new_result
        self.time_step += 1
        return self.turn_results[index]
