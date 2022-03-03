import torch

from .log_squish import LogSquish


class Actor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.distribution_parameters = torch.nn.Parameter(torch.randn([4]))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return torch.tile(self.distribution_parameters, [observations.shape[0], 1])
