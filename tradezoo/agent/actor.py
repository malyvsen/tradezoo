import torch

from .decision_batch import DecisionBatch
from .state_batch import StateBatch


class Actor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(5, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(16, 4),
        )

    def forward(self, state_batch: StateBatch) -> DecisionBatch:
        decision_parameters = self.network(state_batch.tensor)
        return DecisionBatch(
            ask_means=decision_parameters[:, 0],
            ask_scales=decision_parameters[:, 1],
            bid_means=decision_parameters[:, 2],
            bid_scales=decision_parameters[:, 3],
        )
