from dataclasses import dataclass

from tradezoo.agent import Agent, Decision
from tradezoo.market import Account, Market, BuyOrder, SellOrder
from .client import Client
from .state import State


@dataclass
class Trader:
    """A market participant, along with a client."""

    agent: Agent
    account: Account
    client: Client

    def state(self, market: Market):
        return State(
            cash_balance=self.account.cash_balance,
            asset_balance=self.account.asset_balance,
            best_ask=market.best_ask(visible_to=self.account),
            best_bid=market.best_bid(visible_to=self.account),
        )

    def order(self, state: State, decision: Decision):
        asset_allocation_change = (
            decision.target_asset_allocation - state.asset_allocation
        )
        asset_value_change = asset_allocation_change * state.net_worth
        asset_balance_change = asset_value_change / state.mid_price
        if asset_balance_change < 0:
            return SellOrder.public(
                submitted_by=self.account,
                price=state.best_bid / (1 + decision.desperation),
                volume=-asset_balance_change,
            )
        return BuyOrder.public(
            submitted_by=self.trader.account,
            price=state.best_ask * (1 + decision.desperation),
            volume=asset_balance_change,
        )
