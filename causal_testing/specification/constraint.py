from abc import ABC, abstractmethod
import numpy as np

class Constraint(ABC):

    @abstractmethod
    def sample_value(self):
        ...


class NormalDistribution(Constraint):

    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def sample_value(self):
        return np.random.normal(self.mean, self.var)

    def __str__(self):
        return f'N[{self.mean}, {self.var}]'

    def __repr__(self):
        return f'~ N[{self.mean}, {self.var}]'


class AbsoluteValue(Constraint):

    def __init__(self, value):
        self.value = value

    def sample_value(self):
        return self.value

    def __str__(self):
        return f'=={self.value}'

    def __repr__(self):
        return f'=={self.value}'


class UniformDistribution(Constraint):
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def sample_value(self):
        return np.random.uniform(min, max)

    def __str__(self):
        return f'U[{self.min}, {self.max}]'

    def __repr__(self):
        return f'~ U[{self.min}, {self.max}]'
