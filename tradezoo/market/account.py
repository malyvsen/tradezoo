from dataclasses import dataclass


@dataclass
class Account:
    cash_balance: float
    asset_balance: float

    def net_worth(self, asset_value):
        return self.cash_balance + asset_value * self.asset_balance
