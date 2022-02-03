from dataclasses import dataclass
import numpy as np


@dataclass
class GeometricBrownianMotion:
    value: float
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
