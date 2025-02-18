"""This module contains the Variable abstract class, as well as its concrete extensions: Input, Output and Meta."""

from __future__ import annotations
from abc import ABC
from collections.abc import Callable
from typing import TypeVar

import lhsmdu
from pandas import DataFrame
from scipy.stats._distn_infrastructure import rv_generic

# Declare type variable
T = TypeVar("T")


class Variable(ABC):
    """An abstract class representing causal variables.

    :param str name: The name of the variable.
    :param T datatype: The datatype of the variable.
    :param rv_generic distribution: The expected distribution of the variable values.
    :attr name:
    :attr datatype:
    :attr distribution:
    :attr hidden:
    """

    name: str
    datatype: T
    distribution: rv_generic

    def __init__(self, name: str, datatype: T, distribution: rv_generic = None, hidden: bool = False):
        self.name = name
        self.datatype = datatype
        self.distribution = distribution
        self.hidden = hidden

    def __repr__(self):
        return f"{self.typestring()}: {self.name}::{self.datatype.__name__}"

    def sample(self, n_samples: int) -> [T]:
        """Generate a Latin Hypercube Sample of size n_samples according to the
        Variable's distribution.

        :param int n_samples: The number of samples to generate.
        :return: A list of samples
        :rtype: List[T]

        """
        assert self.distribution is not None, "Sampling requires a distribution to be specified."
        lhs = lhsmdu.sample(1, n_samples).tolist()[0]
        return lhsmdu.inverseTransformSample(self.distribution, lhs).tolist()

    def typestring(self) -> str:
        """Return the type of the Variable, e.g. INPUT, or OUTPUT. Note that
        this is NOT the datatype (int, str, etc.).

        :return: A string representing the variable Type.
        :rtype: str

        """
        return type(self).__name__

    def copy(self, name: str = None) -> Variable:
        """Return a new instance of the Variable with the given name, or with
        the original name if no name is supplied.

        :param str name: The variable name.
        :return: A new Variable instance.
        :rtype: Variable

        """
        if name:
            return self.__class__(name, self.datatype, self.distribution)
        return self.__class__(self.name, self.datatype, self.distribution)


class Input(Variable):
    """An extension of the Variable class representing inputs."""


class Output(Variable):
    """An extension of the Variable class representing outputs."""


class Meta(Variable):
    """An extension of the Variable class representing metavariables. These are variables which are relevant to the
    _causal_ structure and properties we may want to test, but are not directly related to the computational model
    either as inputs or outputs.

    :param str name: The name of the variable.
    :param T datatype: The datatype of the variable.
    :param Callable[[DataFrame], DataFrame] populate: Populate a given dataframe containing runtime data with the
    metavariable values as calculated from model inputs and ouputs.
    :attr populate:

    """

    populate: Callable[[DataFrame], DataFrame]

    def __init__(self, name: str, datatype: T, populate: Callable[[DataFrame], DataFrame]):
        super().__init__(name, datatype)
        self.populate = populate
