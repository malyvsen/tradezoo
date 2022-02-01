import torch

from .state_batch import StateBatch


class Critic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(5, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(16, 1),
        )

    def forward(self, state_batch: StateBatch) -> torch.Tensor:
        return self.network(torch.from_numpy(state_batch.tensor)).squeeze(1)
