from dataclasses import dataclass
import torch
from typing import Dict

from tradezoo.agent import Agent, ObservationBatch
from tradezoo.game import Game, TurnResult
from .replay_buffer import ReplayBuffer


@dataclass
class Trainer:
    game: Game
    replay_buffers: Dict[Agent, ReplayBuffer]

    @classmethod
    def new(cls, game: Game, replay_buffer_capacity: int) -> "Trainer":
        return cls(
            game=game,
            replay_buffers={
                trader.agent: ReplayBuffer.empty(capacity=replay_buffer_capacity)
                for trader in game.traders
            },
        )

    def turn_(self) -> TurnResult:
        turn_result = self.game.turn_()
        self.replay_buffers[turn_result.trader.agent].register_turn_(turn_result)
        return turn_result

    def train_step_(self, agent: Agent, batch_size: int):
        experiences = self.replay_buffers[agent].sample(batch_size)
        old_observations = ObservationBatch(
            observations=[experience.old_observation for experience in experiences]
        )
        actions = [experience.action for experience in experiences]
        rewards = torch.tensor([experience.reward for experience in experiences])
        new_observations = ObservationBatch(
            observations=[experience.new_observation for experience in experiences]
        )
        agent.train_step_(
            old_observations=old_observations,
            actions=actions,
            rewards=rewards,
            new_observations=new_observations,
        )
