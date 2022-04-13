from dataclasses import dataclass
from functools import cache
import math
import numpy as np
from typing import List, Union


@dataclass(frozen=True)
class TimeSeries:
    def value(self, step: int) -> float:
        raise NotImplementedError()

    def exp(self):
        return ExpTimeSeries(self)

    def __neg__(self):
        return 0 - self

    def __add__(self, other: Union["TimeSeries", float]) -> "TimeSeries":
        return TimeSeriesSum(
            [self, other if isinstance(other, TimeSeries) else Constant(other)]
        )

    def __radd__(self, other: float):
        return self.__add__(other)

    def __sub__(self, other: Union["TimeSeries", float]) -> "TimeSeries":
        return self + (-other)

    def __rsub__(self, other: float) -> "TimeSeries":
        return -self + other

    def __mul__(self, other: Union["TimeSeries", float]) -> "TimeSeries":
        return TimeSeriesProduct(
            [self, other if isinstance(other, TimeSeries) else Constant(other)]
        )

    def __rmul__(self, other: float):
        return self.__mul__(other)

    def __truediv__(self, other: float) -> "TimeSeries":
        return self * (1 / other)


@dataclass(frozen=True)
class TimeSeriesSum(TimeSeries):
    components: List[TimeSeries]

    def value(self, step: int) -> float:
        return sum(component.value(step) for component in self.components)


@dataclass(frozen=True)
class TimeSeriesProduct(TimeSeries):
    components: List[TimeSeries]

    def value(self, step: int) -> float:
        return np.prod([component.value(step) for component in self.components])


@dataclass(frozen=True)
class ExpTimeSeries(TimeSeries):
    slave: TimeSeries

    def value(self, step: int) -> float:
        return math.exp(self.slave.value(step))


@dataclass(frozen=True)
class Constant(TimeSeries):
    initial_value: float

    def value(self, step: int) -> float:
        return self.initial_value


@dataclass(frozen=True)
class SineWave(TimeSeries):
    period: float

    def value(self, step: int) -> float:
        return np.sin(step * 2 * np.pi / self.period)


@dataclass(frozen=True)
class GeometricBrownianMotion(TimeSeries):
    initial_value: float
    underlying_mean: float
    underlying_std: float

    @classmethod
    def driftless(
        cls, initial_value: float, underlying_std: float
    ) -> "GeometricBrownianMotion":
        """Geometric Brownian motion whose expected value is constant over time"""
        return cls(
            initial_value=initial_value,
            underlying_mean=-(underlying_std ** 2) / 2,
            underlying_std=underlying_std,
        )

    @cache
    def value(self, step: int) -> float:
        assert step >= 0
        if step == 0:
            return self.initial_value
        return self.value(step - 1) * np.random.lognormal(
            mean=self.underlying_mean, sigma=self.underlying_std
        )
