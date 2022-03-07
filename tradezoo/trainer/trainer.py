from dataclasses import dataclass
import torch
from typing import Dict, List

from tradezoo.agent import Agent, ObservationBatch
from tradezoo.game import Game, TurnResult
from .experience import Experience
from .replay_buffer import ReplayBuffer
from .train_result import TrainResult


@dataclass
class Trainer:
    game: Game
    replay_buffers: Dict[Agent, ReplayBuffer]
    batch_size: int

    @classmethod
    def new(cls, game: Game, replay_buffer_capacity: int, batch_size: int) -> "Trainer":
        return cls(
            game=game,
            replay_buffers={
                trader.agent: ReplayBuffer.empty(capacity=replay_buffer_capacity)
                for trader in game.traders
            },
            batch_size=batch_size,
        )

    def turn_(self) -> TurnResult:
        turn_result = self.game.turn_()
        replay_buffer = self.replay_buffers[turn_result.trader.agent]
        replay_buffer.register_turn_(turn_result)
        if len(replay_buffer.experiences) >= self.batch_size:
            self.train_(
                agent=turn_result.trader.agent,
                experiences=replay_buffer.sample(self.batch_size),
            )
        return turn_result

    @classmethod
    def train_(cls, agent: Agent, experiences: List[Experience]):
        old_observations = ObservationBatch(
            observations=[experience.old_observation for experience in experiences]
        )
        actions = [experience.action for experience in experiences]
        rewards = torch.tensor([experience.reward for experience in experiences])
        new_observations = ObservationBatch(
            observations=[experience.new_observation for experience in experiences]
        )

        td_error = (
            rewards
            + agent.discount_factor * agent.evaluate(new_observations).detach()
            - agent.evaluate(old_observations)
        )

        critic_loss = td_error.square().mean()
        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        agent.critic_optimizer.step()

        actor_loss = (
            -td_error.detach()
            * agent.decide(old_observations).log_probabilities(actions)
        ).mean()
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        agent.actor_optimizer.step()

        return TrainResult(actor_loss=actor_loss.item(), critic_loss=critic_loss.item())
