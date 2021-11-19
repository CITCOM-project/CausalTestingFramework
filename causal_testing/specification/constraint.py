from abc import ABC, abstractmethod
from typing import Union
import numpy as np


class Constraint(ABC):
    """
    A constraint is a restriction that can be assigned to a subset of inputs that characterises a particular usage
    scenario for the system-under-test.
    """
    @abstractmethod
    def sample_value(self) -> Union[int, float, str]:
        """
        Sample a value from the specified constraint (absolute value or some distribution).
        :return: A value sampled at random from the applied constraint (absolute value or some distribution).
        """
        pass


class NormalDistribution(Constraint):
    """
    A normal distribution which can be used to constrain the values of inputs in a Scenario.
    """

    def __init__(self, mean: Union[int, float], var: Union[int, float]):
        self.mean = mean
        self.var = var

    def sample_value(self) -> float:
        """
        Sample a value at random from the normal distribution.
        :return: A randomly sampled value from the normal distribution.
        """
        return np.random.normal(self.mean, self.var)

    def __str__(self):
        return f'N[{self.mean}, {self.var}]'

    def __repr__(self):
        return f'~ N[{self.mean}, {self.var}]'


class AbsoluteValue(Constraint):

    def __init__(self, value: Union[int, float, str]):
        self.value = value

    def sample_value(self) -> Union[int, float, str]:
        """
        Sample the value of the absolute value constraint.
        :return: The value of the absolute value constraint.
        """
        return self.value

    def __str__(self):
        return f'=={self.value}'

    def __repr__(self):
        return f'=={self.value}'


class UniformDistribution(Constraint):
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def sample_value(self) -> float:
        """
        Sample a value from the uniform distribution.
        :return: A float sampled at random from the uniform distribution.
        """
        return np.random.uniform(self.lower, self.upper)

    def __str__(self):
        return f'U[{self.lower}, {self.upper}]'

    def __repr__(self):
        return f'~ U[{self.lower}, {self.upper}]'
