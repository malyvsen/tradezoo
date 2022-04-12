from dataclasses import dataclass
from typing import List
from tqdm.auto import trange

from tradezoo.agent import Agent
from tradezoo.game import Game, Trader, TurnResult
from .plots import balance_plot, reward_plot, trades_plot, training_plot
from .learning_agent import LearningAgent
from .train_result import TrainResult


@dataclass(frozen=True)
class Experiment:
    @dataclass(frozen=True)
    class TimeStep:
        turn_result: TurnResult
        train_results: List[TrainResult]

    game: Game
    time_steps: List[TimeStep]

    @classmethod
    def run_(cls, game: Game, num_steps: int):
        time_steps = []
        for step in trange(num_steps):
            turn_result = game.turn_()
            agent = turn_result.trader.agent
            time_steps.append(
                cls.TimeStep(
                    turn_result=turn_result,
                    train_results=agent.post_turn_(turn_result)
                    if isinstance(agent, LearningAgent)
                    else [],
                )
            )
        return cls(game=game, time_steps=time_steps)

    def balance_plot(self, trader: Trader):
        return balance_plot(self.turn_results(trader))

    def reward_plot(self, trader: Trader):
        return reward_plot(agent=trader.agent, turn_results=self.turn_results(trader))

    def trades_plot(self, trader: Trader):
        return trades_plot(self.turn_results(trader))

    def training_plot(self, agent: Agent):
        return training_plot(self.train_results(agent))

    def turn_results(self, trader: Trader):
        return [
            time_step.turn_result
            for time_step in self.time_steps
            if time_step.turn_result.trader is trader
        ]

    def train_results(self, agent: Agent):
        return [
            train_result
            for time_step in self.time_steps
            if time_step.turn_result.trader.agent is agent
            for train_result in time_step.train_results
        ]
