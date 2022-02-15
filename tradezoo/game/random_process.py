from dataclasses import dataclass
import numpy as np
from typing import List


@dataclass
class RandomProcess:
    value: float

    def step_(self):
        raise NotImplementedError()


@dataclass
class ProductRandomProcess(RandomProcess):
    components: List[RandomProcess]

    @property
    def value(self) -> float:
        return np.prod([component.value for component in self.components])

    def step_(self):
        for component in self.components:
            component.step_()


@dataclass
class Constant(RandomProcess):
    def step_(self):
        pass


@dataclass
class GeometricBrownianMotion(RandomProcess):
    underlying_mean: float
    underlying_std: float

    @classmethod
    def driftless(
        cls, initial_value: float, underlying_std: float
    ) -> "GeometricBrownianMotion":
        """Geometric Brownian motion whose expected value is constant over time"""
        return cls(
            value=initial_value,
            underlying_mean=-(underlying_std ** 2) / 2,
            underlying_std=underlying_std,
        )

    def step_(self):
        self.value *= np.random.lognormal(
            mean=self.underlying_mean, sigma=self.underlying_std
        )
