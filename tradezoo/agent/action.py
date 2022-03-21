from dataclasses import dataclass


@dataclass(frozen=True)
class Action:
    asset_allocation: float
    """The fraction of the total portfolio allocated to the asset"""

    @property
    def constrained_asset_allocation(self):
        """The asset allocation assuming no shorting/leverage"""
        return min(max(self.asset_allocation, 0), 1)
