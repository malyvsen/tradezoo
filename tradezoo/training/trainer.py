from dataclasses import dataclass
import torch
from typing import Dict

from .replay_buffer import ReplayBuffer
from tradezoo.agent import ObservationBatch
from tradezoo.game import Game, TurnResult
from tradezoo.trader import Trader


@dataclass
class Trainer:
    game: Game
    replay_buffers: Dict[Trader, ReplayBuffer]

    @classmethod
    def new(cls, game: Game, replay_buffer_capacity: int) -> "Trainer":
        return cls(
            game=game,
            replay_buffers={
                trader: ReplayBuffer.empty(capacity=replay_buffer_capacity)
                for trader in game.traders
            },
        )

    @property
    def traders(self):
        return self.game.traders

    def turn_(self) -> TurnResult:
        turn_result = self.game.turn_()
        self.replay_buffers[turn_result.trader].register_turn_(turn_result)
        return turn_result

    def train_step_(self, trader: Trader, batch_size: int):
        experiences = self.replay_buffers[trader].sample(batch_size)
        old_observations = ObservationBatch(
            observations=[experience.old_observation for experience in experiences]
        )
        actions = [experience.action for experience in experiences]
        rewards = torch.tensor([experience.reward for experience in experiences])
        new_observations = ObservationBatch(
            observations=[experience.new_observation for experience in experiences]
        )
        trader.agent.train_step_(
            old_observations=old_observations,
            actions=actions,
            rewards=rewards,
            new_observations=new_observations,
        )
