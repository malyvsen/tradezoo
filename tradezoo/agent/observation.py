from dataclasses import dataclass
from functools import cached_property
import math
import numpy as np
import torch
from typing import List


@dataclass(frozen=True)
class Observation:
    asset_allocation: float
    best_ask: float
    best_bid: float

    @property
    def array(self):
        return np.array(
            [self.asset_allocation, math.log(self.best_ask), math.log(self.best_bid)],
            dtype=np.float32,
        )


@dataclass(frozen=True)
class ObservationSeries:
    observations: List[Observation]

    @property
    def batch(self):
        return ObservationSeriesBatch(series=[self])

    @cached_property
    def array(self):
        return np.stack(
            [observation.array for observation in self.observations],
            axis=0,
        )

    def with_new(self, observation: Observation, horizon: int):
        """A distinct ObservationSeries, with the given observation at the end"""
        return type(self)(observations=(self.observations + [observation])[-horizon:])


@dataclass(frozen=True)
class ObservationSeriesBatch:
    series: List[ObservationSeries]

    @property
    def tensor(self) -> torch.Tensor:
        return torch.from_numpy(
            np.stack(
                [series.array for series in self.series],
                axis=0,
            )
        )
