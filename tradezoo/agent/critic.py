import torch


class Critic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(4, 16),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(16, 1),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations).squeeze(1)
