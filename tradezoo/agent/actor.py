import torch

from .log_squish import LogSquish


class Actor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(4, 16),
            LogSquish(),
            torch.nn.Linear(16, 4),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)
