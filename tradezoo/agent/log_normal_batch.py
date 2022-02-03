from dataclasses import dataclass
from functools import cached_property
import torch


@dataclass(frozen=True)
class LogNormalBatch:
    underlying_means: torch.Tensor
    underlying_stds: torch.Tensor

    @cached_property
    def torch_distribution(self) -> torch.distributions.LogNormal:
        return torch.distributions.LogNormal(
            self.underlying_means, self.underlying_stds
        )

    def sample(self) -> torch.Tensor:
        return self.torch_distribution.sample()

    def log_probabilities(self, value: torch.Tensor) -> torch.Tensor:
        return self.torch_distribution.log_prob(value)
