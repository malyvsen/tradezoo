from dataclasses import dataclass, asdict
import json
import math
import numpy as np
import torch
from tqdm.auto import tqdm
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
    trader_account = Account(cash_balance=2048, asset_balance=2048)
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
    return Experiment.run_(game=game, num_steps=1024, loading_bar=False)


def total_balances(hyperparameters, num_runs=4):
    experiments = [run_experiment(hyperparameters) for _ in range(num_runs)]
    final_states = [
        experiment.time_steps[-1].turn_result.state for experiment in experiments
    ]
    return [state.cash_balance + state.asset_balance for state in final_states]


def main():
    all_hyperparameters = [
        Hyperparameters(
            exploration_level=exploration_level,
            discount_factor=discount_factor,
            replay_buffer_capacity=replay_buffer_capacity,
            batch_size=batch_size,
            train_steps_per_turn=train_steps_per_turn,
            learning_rate=learning_rate,
            steps_per_target_update=steps_per_target_update,
        )
        for exploration_level in [256, 1024, 4096]
        for discount_factor in [0.9, 0.99]
        for replay_buffer_capacity in [16, 64]
        for batch_size in [1, 16]
        for train_steps_per_turn in [16, 64]
        for learning_rate in [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3]
        for steps_per_target_update in [256, 1024, 4096]
    ]
    with open("./hyperparameter_results.json", "w") as save_file:
        save_file.write("[\n")
        for hyperparameters in tqdm(all_hyperparameters):
            result_dict = dict(
                hyperparameters=asdict(hyperparameters),
                total_balances=total_balances(hyperparameters),
            )
            save_file.write(f"{json.dumps(result_dict)},\n")
            save_file.flush()
        save_file.write("]\n")


main()
