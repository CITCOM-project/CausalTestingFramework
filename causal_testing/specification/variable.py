from __future__ import annotations
from pandas import DataFrame
from typing import Callable, TypeVar
from scipy.stats._distn_infrastructure import rv_generic
from z3 import ExprRef, Context, Int, String, Real
from abc import ABC, abstractmethod
import lhsmdu

# Declare type variable
# Is there a better way? I'd really like to do Variable[T](ExprRef)
T = TypeVar("T")

z3_types = {int: Int, str: String, float: Real}


def _coerce(val: any) -> any:
    """Coerce Variables to their Z3 equivalents if appropriate to do so,
    otherwise assume literal constants.

    :param any val: A value, possibly a Variable.
    :return: Either a Z3 ExprRef representing the variable or the original value.
    :rtype: any

    """
    if isinstance(val, Variable):
        return val.z3
    return val


class Variable(ABC):
    name: str
    datatype: T
    distribution: rv_generic

    def __init__(self, name: str, datatype: T, distribution: rv_generic = None):
        self.name = name
        self.datatype = datatype
        self.z3 = z3_types[datatype](name)
        self.distribution = distribution

    def __repr__(self):
        return f"{self.typestring()}: {self.name}::{self.datatype}"

    # TODO: We're going to need to implement all the supported Z3 operations like this
    def __ge__(self, other: any) -> BoolRef:
        """Create the Z3 expression `other >= self`.

        :param any other: The object to compare against.
        :return: The Z3 expression `other >= self`.
        :rtype: BoolRef
        """
        return self.z3.__ge__(_coerce(other))

    def __le__(self, other: any) -> BoolRef:
        """Create the Z3 expression `other <= self`.

        :param any other: The object to compare against.
        :return: The Z3 expression `other >= self`.
        :rtype: BoolRef
        """
        return self.z3.__le__(_coerce(other))

    def __gt__(self, other: any) -> BoolRef:
        """Create the Z3 expression `other > self`.

        :param any other: The object to compare against.
        :return: The Z3 expression `other >= self`.
        :rtype: BoolRef
        """
        return self.z3.__gt__(_coerce(other))

    def __lt__(self, other: any) -> BoolRef:
        """Create the Z3 expression `other < self`.

        :param any other: The object to compare against.
        :return: The Z3 expression `other >= self`.
        :rtype: BoolRef
        """
        print(self, other)
        return self.z3.__lt__(_coerce(other))

    def cast(val: any) -> T:
        """Cast the supplied value to the datatype T of the variable.

        :param any val: The value to cast.
        :return: The supplied value as an instance of T.
        :rtype: T
        """
        return datatype(t)

    def sample(n_samples: int) -> [T]:
        """Generate a Latin Hypercube Sample of size n_samples according to the
        Variable's distribution.

        :param int n_samples: The number of samples to generate.
        :return: A list of samples
        :rtype: List[T]

        """
        assert (
            self.distribution is not None
        ), "Sampling requires a distribution to be specified."
        lhs = lhsmdu.sample(1, n_samples).tolist()[0]
        return lhsmdu.inverseTransformSample(self.distribution, lhs).tolist()[0]

    @abstractmethod
    def typestring(self) -> str:
        """Return the type of the Variable, e.g. INPUT, or OUTPUT. Note that
        this is NOT the datatype (int, str, etc.).

        :return: A string representing the variable Type.
        :rtype: str

        """
        pass

    @abstractmethod
    def copy(self, name: str = None) -> Variable:
        """Return a new instance of the Variable with the given name, or with
        the original name if no name is supplied.

        :param str name: The variable name.
        :return: A new Variable instance.
        :rtype: Variable

        """
        pass


class Input(Variable):
    def typestring(self) -> str:
        return "INPUT"

    def copy(self, name=None) -> Input:
        if name:
            return Input(name, self.datatype, self.distribution)
        return Input(self.name, self.datatype, self.distribution)


class Output(Variable):
    def typestring(self) -> str:
        return "OUTPUT"

    def copy(self, name=None) -> Output:
        if name:
            return Output(name, self.datatype, self.distribution)
        return Output(self.name, self.datatype, self.distribution)


class Meta(Variable):
    populate: Callable[[DataFrame], DataFrame]

    def __init__(
        self, name: str, datatype: T, populate: Callable[[DataFrame], DataFrame]
    ):
        super().__init__(name, datatype)
        self.populate = populate

    def typestring(self) -> str:
        return "META"

    def copy(self, name=None) -> Meta:
        if name:
            return Meta(name, self.datatype, self.distribution)
        return Meta(self.name, self.datatype, self.distribution)
