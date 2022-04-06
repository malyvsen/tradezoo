from dataclasses import dataclass
from functools import cached_property
import numpy as np
import torch
from typing import List


@dataclass(frozen=True)
class Observation:
    asset_allocation: float
    best_ask: float
    best_bid: float


@dataclass(frozen=True)
class ObservationSeries:
    observations: List[Observation]

    @property
    def batch(self):
        return ObservationSeriesBatch(series=[self])

    @cached_property
    def array(self):
        def gather(attribute: str):
            return np.array(
                [getattr(observation, attribute) for observation in self.observations],
                dtype=np.float32,
            )

        return np.stack(
            [
                gather("asset_allocation"),
                np.log(gather("best_bid")),
                np.log(gather("best_ask")),
            ],
            axis=0,
        )

    def with_new(self, observation: Observation, horizon: int):
        """A distinct ObservationSeries, with the given observation at the end"""
        return type(self)(observation=(self.observations + [observation])[-horizon:])


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
