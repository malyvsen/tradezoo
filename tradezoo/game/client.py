from dataclasses import dataclass
from typing import List

from tradezoo.market import Account, Order, BuyOrder, SellOrder
from .random_process import RandomProcess


@dataclass
class Client:
    account: Account
    ask_process: RandomProcess
    bid_process: RandomProcess

    @classmethod
    def inexhaustible(cls, ask_process: RandomProcess, bid_process: RandomProcess):
        return cls(
            account=Account(cash_balance=float("inf"), asset_balance=float("inf")),
            ask_process=ask_process,
            bid_process=bid_process,
        )

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
