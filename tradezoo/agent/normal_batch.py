from dataclasses import dataclass
from functools import cached_property
import torch


@dataclass(frozen=True)
class NormalBatch:
    means: torch.Tensor
    scales: torch.Tensor

    @cached_property
    def torch_distribution(self) -> torch.distributions.Normal:
        return torch.distributions.Normal(self.means, self.scales)

    def sample(self) -> torch.Tensor:
        return self.torch_distribution.sample()

    def log_probabilities(self, value: torch.Tensor) -> torch.Tensor:
        return self.torch_distribution.log_prob(value)
