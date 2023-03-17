"""This module contains the Variable abstract class, as well as its concrete extensions: Input, Output and Meta. The
function z3_types and the private function _coerce are also in this module."""

from __future__ import annotations
from abc import ABC
from collections.abc import Callable
from enum import Enum
from typing import Any, TypeVar

import lhsmdu
from pandas import DataFrame
from scipy.stats._distn_infrastructure import rv_generic
from z3 import Bool, BoolRef, Const, EnumSort, Int, RatNumRef, Real, String

# Declare type variable
T = TypeVar("T")
z3 = TypeVar("Z3")


def z3_types(datatype: T) -> z3:
    """Cast datatype to Z3 datatype
    :param datatype: python datatype to be cast
    :return: Type name compatible with Z3 library
    """
    types = {int: Int, str: String, float: Real, bool: Bool}
    if datatype in types:
        return types[datatype]
    if issubclass(datatype, Enum):
        dtype, _ = EnumSort(datatype.__name__, [str(x.value) for x in datatype])
        return lambda x: Const(x, dtype)
    if hasattr(datatype, "to_z3"):
        return datatype.to_z3()
    raise ValueError(
        f"Cannot convert type {datatype} to Z3."
        + " Please use a native type, an Enum, or implement a conversion manually."
    )


def _coerce(val: Any) -> Any:
    """Coerce Variables to their Z3 equivalents if appropriate to do so,
    otherwise assume literal constants.

    :param any val: A value, possibly a Variable.
    :return: Either a Z3 ExprRef representing the variable or the original value.
    :rtype: Any

    """
    if isinstance(val, Variable):
        return val.z3
    return val


class Variable(ABC):
    """An abstract class representing causal variables.

    :param str name: The name of the variable.
    :param T datatype: The datatype of the variable.
    :param rv_generic distribution: The expected distribution of the variable values.
    :attr type z3: The Z3 mirror of the variable.
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
        self.z3 = z3_types(datatype)(name)
        self.distribution = distribution
        self.hidden = hidden

    def __repr__(self):
        return f"{self.typestring()}: {self.name}::{self.datatype.__name__}"

    # Thin wrapper for Z1 functions

    def __add__(self, other: Any) -> BoolRef:
        """Create the Z3 expression `self + other`.

        :param any other: The object to compare against.
        :return: The Z3 expression `self + other`.
        :rtype: BoolRef
        """
        return self.z3.__add__(_coerce(other))

    def __ge__(self, other: Any) -> BoolRef:
        """Create the Z3 expression `self >= other`.

        :param any other: The object to compare against.
        :return: The Z3 expression `self >= other`.
        :rtype: BoolRef
        """
        return self.z3.__ge__(_coerce(other))

    def __gt__(self, other: Any) -> BoolRef:
        """Create the Z3 expression `self > other`.

        :param any other: The object to compare against.
        :return: The Z3 expression `self > other`.
        :rtype: BoolRef
        """
        return self.z3.__gt__(_coerce(other))

    def __le__(self, other: Any) -> BoolRef:
        """Create the Z3 expression `self <= other`.

        :param any other: The object to compare against.
        :return: The Z3 expression `self <= other`.
        :rtype: BoolRef
        """
        return self.z3.__le__(_coerce(other))

    def __lt__(self, other: Any) -> BoolRef:
        """Create the Z3 expression `self < other`.

        :param any other: The object to compare against.
        :return: The Z3 expression `self < other`.
        :rtype: BoolRef
        """
        return self.z3.__lt__(_coerce(other))

    def __mod__(self, other: Any) -> BoolRef:
        """Create the Z3 expression `self % other`.

        :param any other: The object to compare against.
        :return: The Z3 expression `self % other`.
        :rtype: BoolRef
        """
        return self.z3.__mod__(_coerce(other))

    def __mul__(self, other: Any) -> BoolRef:
        """Create the Z3 expression `self * other`.

        :param any other: The object to compare against.
        :return: The Z3 expression `self * other`.
        :rtype: BoolRef
        """
        return self.z3.__mul__(_coerce(other))

    def __ne__(self, other: Any) -> BoolRef:
        """Create the Z3 expression `self != other`.

        :param any other: The object to compare against.
        :return: The Z3 expression `self != other`.
        :rtype: BoolRef
        """
        return self.z3.__ne__(_coerce(other))

    def __neg__(self) -> BoolRef:
        """Create the Z3 expression `-self`.

        :param any other: The object to compare against.
        :return: The Z3 expression `-self`.
        :rtype: BoolRef
        """
        return self.z3.__neg__()

    def __pow__(self, other: Any) -> BoolRef:
        """Create the Z3 expression `self ^ other`.

        :param any other: The object to compare against.
        :return: The Z3 expression `self ^ other`.
        :rtype: BoolRef
        """
        return self.z3.__pow__(_coerce(other))

    def __sub__(self, other: Any) -> BoolRef:
        """Create the Z3 expression `self - other`.

        :param any other: The object to compare against.
        :return: The Z3 expression `self - other`.
        :rtype: BoolRef
        """
        return self.z3.__sub__(_coerce(other))

    def __truediv__(self, other: Any) -> BoolRef:
        """Create the Z3 expression `self / other`.

        :param any other: The object to compare against.
        :return: The Z3 expression `self / other`.
        :rtype: BoolRef
        """
        return self.z3.__truediv__(_coerce(other))

    # End thin wrapper

    def cast(self, val: Any) -> T:
        """Cast the supplied value to the datatype T of the variable.

        :param any val: The value to cast.
        :return: The supplied value as an instance of T.
        :rtype: T
        """
        assert val is not None, f"Invalid value None for variable {self}"
        if isinstance(val, self.datatype):
            return val
        if isinstance(val, BoolRef) and self.datatype == bool:
            return str(val) == "True"
        if isinstance(val, RatNumRef) and self.datatype == float:
            return float(val.numerator().as_long() / val.denominator().as_long())
        if hasattr(val, "is_string_value") and val.is_string_value() and self.datatype == str:
            return val.as_string()
        if (isinstance(val, (float, int, bool))) and (self.datatype in (float, int, bool)):
            return self.datatype(val)
        return self.datatype(str(val))

    def z3_val(self, z3_var, val: Any) -> T:
        """Cast value to Z3 value"""
        native_val = self.cast(val)
        if isinstance(native_val, Enum):
            values = [z3_var.sort().constructor(c)() for c in range(z3_var.sort().num_constructors())]
            values = [v for v in values if val.__class__(str(v)) == val]
            assert len(values) == 1, f"Expected {values} to be length 1"
            return values[0]
        return native_val

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
