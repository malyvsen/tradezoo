from dataclasses import dataclass
from functools import cached_property
import torch


@dataclass(frozen=True)
class LogNormalBatch:
    underlying_means: torch.Tensor
    underlying_stds: torch.Tensor

    @cached_property
    def means(self) -> torch.Tensor:
        return torch.exp(self.underlying_means + self.underlying_stds ** 2 / 2)

    @cached_property
    def torch_distribution(self) -> torch.distributions.LogNormal:
        return torch.distributions.LogNormal(
            loc=self.underlying_means, scale=self.underlying_stds, validate_args=True
        )

    def sample(self) -> torch.Tensor:
        return self.torch_distribution.sample()

    def log_probabilities(self, value: torch.Tensor) -> torch.Tensor:
        return self.torch_distribution.log_prob(value)
