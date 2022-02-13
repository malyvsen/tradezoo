from dataclasses import dataclass
from typing import List

from tradezoo.market import Account, Order, BuyOrder, SellOrder
from .random_process import RandomProcess


@dataclass
class MarketMaker:
    account: Account
    ask_process: RandomProcess
    bid_process: RandomProcess

    def orders(self) -> List[Order]:
        return [
            SellOrder(
                submitted_by=self.account,
                price=self.ask_process.value,
                volume=float("inf"),
            ),
            BuyOrder(
                submitted_by=self.account,
                price=self.bid_process.value,
                volume=float("inf"),
            ),
        ]
