from dataclasses import dataclass


@dataclass
class Account:
    cash_balance: float
    stock_balance: float

    def net_worth(self, stock_value):
        return self.cash_balance + stock_value * self.stock_balance
