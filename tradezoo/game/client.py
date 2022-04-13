from dataclasses import dataclass
from typing import List

from tradezoo.market import Account, Order, BuyOrder, SellOrder, WhitelistFilter
from .time_series import TimeSeries


@dataclass
class Client:
    """
    An entity used to motivate trading.
    Does not itself participate in the market except for interacting with a single trader.
    """

    account: Account
    for_account: Account
    ask_process: TimeSeries
    bid_process: TimeSeries

    def orders(self, time_step: int) -> List[Order]:
        return [
            SellOrder(
                submitted_by=self.account,
                visibility=WhitelistFilter([self.for_account]),
                price=self.ask_process.value(time_step),
                volume=float("inf"),
            ),
            BuyOrder(
                submitted_by=self.account,
                visibility=WhitelistFilter([self.for_account]),
                price=self.bid_process.value(time_step),
                volume=float("inf"),
            ),
        ]
