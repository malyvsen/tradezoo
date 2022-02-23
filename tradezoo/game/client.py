from dataclasses import dataclass
from typing import List

from tradezoo.market import Account, Order, BuyOrder, SellOrder, WhitelistFilter
from .random_process import RandomProcess


@dataclass
class Client:
    account: Account
    for_account: Account
    ask_process: RandomProcess
    bid_process: RandomProcess

    def orders(self) -> List[Order]:
        return [
            SellOrder(
                submitted_by=self.account,
                visibility=WhitelistFilter([self.for_account]),
                price=self.ask_process.value,
                volume=float("inf"),
            ),
            BuyOrder(
                submitted_by=self.account,
                visibility=WhitelistFilter([self.for_account]),
                price=self.bid_process.value,
                volume=float("inf"),
            ),
        ]
