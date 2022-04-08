from dataclasses import dataclass

from tradezoo.game import TurnResult


@dataclass(frozen=True)
class Experience:
    old_turn_result: TurnResult
    new_turn_result: TurnResult

    def __post_init__(self):
        assert self.old_turn_result.trader == self.new_turn_result.trader
        assert self.old_turn_result.time_step < self.new_turn_result.time_step
