from dataclasses import dataclass


@dataclass(frozen=True)
class Action:
    ask: float
    bid: float
