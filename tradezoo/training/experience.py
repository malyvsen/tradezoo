from dataclasses import dataclass

from tradezoo.game import TurnResult


@dataclass(frozen=True)
class Experience:
    old_turn_result: TurnResult
    new_turn_result: TurnResult

    @property
    def full_length(self):
        return (
            len(self.old_turn_result.observations.observations)
            == len(self.new_turn_result.observations.observations)
            == self.trader.agent.horizon
        )

    @property
    def trader(self):
        return self.old_turn_result.trader

    def __post_init__(self):
        assert self.old_turn_result.trader == self.new_turn_result.trader
        assert self.old_turn_result.time_step < self.new_turn_result.time_step
