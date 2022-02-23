from dataclasses import dataclass
from typing import List

from tradezoo.market import Account, Order, BuyOrder, SellOrder, WhitelistFilter
from .time_series import TimeSeries


@dataclass
class Client:
    account: Account
    for_account: Account
    ask_process: TimeSeries
    bid_process: TimeSeries

    def orders(self, turn_number: int) -> List[Order]:
        return [
            SellOrder(
                submitted_by=self.account,
                visibility=WhitelistFilter([self.for_account]),
                price=self.ask_process.value(turn_number),
                volume=float("inf"),
            ),
            BuyOrder(
                submitted_by=self.account,
                visibility=WhitelistFilter([self.for_account]),
                price=self.bid_process.value(turn_number),
                volume=float("inf"),
            ),
        ]
