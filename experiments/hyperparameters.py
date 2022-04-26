from dataclasses import dataclass, asdict
import json
from manydo import map
import math
import numpy as np
import random
import torch
from tradezoo.agent import Critic
from tradezoo.game import Game, Client, SineWave, Trader
from tradezoo.market import Account, Market
from tradezoo.training import Experiment, LearningAgent, ReplayBuffer


@dataclass(frozen=True)
class Hyperparameters:
    exploration_level: float
    discount_factor: float
    replay_buffer_capacity: int
    batch_size: int
    train_steps_per_turn: int
    learning_rate: float
    steps_per_target_update: int


def make_agent(hyperparameters: Hyperparameters):
    critic = Critic()
    return LearningAgent(
        critic=critic,
        horizon=2,
        allocation_space=np.linspace(0, 1, num=8),
        relative_price_space=2 ** np.linspace(-1, 1, num=64),
        exploration_schedule=lambda step: hyperparameters.exploration_level
        / (step + hyperparameters.exploration_level),
        utility_function=math.log,
        discount_factor=hyperparameters.discount_factor,
        replay_buffer=ReplayBuffer.empty(
            capacity=hyperparameters.replay_buffer_capacity
        ),
        batch_size=hyperparameters.batch_size,
        train_steps_per_turn=hyperparameters.train_steps_per_turn,
        optimizer=torch.optim.Adam(
            critic.parameters(), lr=hyperparameters.learning_rate
        ),
        target=Critic(),
        steps_per_target_update=hyperparameters.steps_per_target_update,
        steps_completed=0,
    )


def run_experiment(hyperparameters):
    trader_account = Account(cash_balance=1, asset_balance=1)
    client_account = Account(cash_balance=float("inf"), asset_balance=float("inf"))
    price_process = 1 + SineWave(period=16) * 0.2
    trader = Trader(
        agent=make_agent(hyperparameters),
        account=trader_account,
        client=Client(
            account=client_account,
            for_account=trader_account,
            ask_process=price_process * 1.1,
            bid_process=price_process * 0.9,
        ),
    )
    game = Game.new(
        market=Market.from_accounts([trader_account, client_account]),
        traders=[trader],
    )
    return Experiment.run_(game=game, num_steps=2048, loading_bar=False)


def total_balances(hyperparameters, num_runs=4):
    experiments = [run_experiment(hyperparameters) for _ in range(num_runs)]
    final_states = [
        experiment.time_steps[-1].turn_result.state for experiment in experiments
    ]
    return [state.cash_balance + state.asset_balance for state in final_states]


def main():
    all_hyperparameters = [
        Hyperparameters(
            exploration_level=2 ** random.randint(0, 12),
            discount_factor=1 - 2 ** random.uniform(-10, -2),
            replay_buffer_capacity=2 ** random.randint(4, 10),
            batch_size=2 ** random.randint(2, 4),
            train_steps_per_turn=2 ** random.randint(6, 8),
            learning_rate=2 ** random.uniform(-15, -9),
            steps_per_target_update=2 ** random.randint(8, 12),
        )
        for _ in range(128)
    ]
    results = map(
        function=lambda hyperparameters: dict(
            hyperparameters=asdict(hyperparameters),
            total_balances=total_balances(hyperparameters),
        ),
        iterable=all_hyperparameters,
        num_jobs=8,
    )
    with open("./hyperparameter_results.json", "w") as save_file:
        json.dump(results, save_file)


main()
