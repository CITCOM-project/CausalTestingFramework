from collections.abc import Callable
from enum import Enum
from pandas import DataFrame
from typing import TypeVar, Any
from scipy.stats._distn_infrastructure import rv_generic
from z3 import Int, String, Real, BoolRef, RatNumRef, Bool, EnumSort, Const
from abc import ABC, abstractmethod
import lhsmdu

# Declare type variable
# Is there a better way? I'd really like to do Variable[T](ExprRef)
T = TypeVar("T")
Variable = TypeVar("Variable")
Input = TypeVar("Input")
Output = TypeVar("Output")
Meta = TypeVar("Meta")


def z3_types(datatype):
    types = {int: Int, str: String, float: Real, bool: Bool}
    if datatype in types:
        return types[datatype]
    if issubclass(datatype, Enum):
        dtype, _ = EnumSort(datatype.__name__, [x.name for x in datatype])
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

    """

    name: str
    datatype: T
    distribution: rv_generic

    def __init__(self, name: str, datatype: T, distribution: rv_generic = None):
        self.name = name
        self.datatype = datatype
        self.z3 = z3_types(datatype)(name)
        self.distribution = distribution

    def __repr__(self):
        return f"{self.typestring()}: {self.name}::{self.datatype.__name__}"

    # TODO: We're going to need to implement all the supported Z3 operations like this
    def __ge__(self, other: Any) -> BoolRef:
        """Create the Z3 expression `other >= self`.

        :param any other: The object to compare against.
        :return: The Z3 expression `other >= self`.
        :rtype: BoolRef
        """
        return self.z3.__ge__(_coerce(other))

    def __le__(self, other: Any) -> BoolRef:
        """Create the Z3 expression `other <= self`.

        :param any other: The object to compare against.
        :return: The Z3 expression `other >= self`.
        :rtype: BoolRef
        """
        return self.z3.__le__(_coerce(other))

    def __gt__(self, other: Any) -> BoolRef:
        """Create the Z3 expression `other > self`.

        :param any other: The object to compare against.
        :return: The Z3 expression `other >= self`.
        :rtype: BoolRef
        """
        return self.z3.__gt__(_coerce(other))

    def __lt__(self, other: Any) -> BoolRef:
        """Create the Z3 expression `other < self`.

        :param any other: The object to compare against.
        :return: The Z3 expression `other >= self`.
        :rtype: BoolRef
        """
        return self.z3.__lt__(_coerce(other))

    def __mul__(self, other: Any) -> BoolRef:
        """Create the Z3 expression `other * self`.

        :param any other: The object to compare against.
        :return: The Z3 expression `other >= self`.
        :rtype: BoolRef
        """
        return self.z3.__mul__(_coerce(other))

    def __sub__(self, other: Any) -> BoolRef:
        """Create the Z3 expression `other * self`.

        :param any other: The object to compare against.
        :return: The Z3 expression `other >= self`.
        :rtype: BoolRef
        """
        return self.z3.__sub__(_coerce(other))

    def __add__(self, other: Any) -> BoolRef:
        """Create the Z3 expression `other * self`.

        :param any other: The object to compare against.
        :return: The Z3 expression `other >= self`.
        :rtype: BoolRef
        """
        return self.z3.__add__(_coerce(other))

    def __truediv__(self, other: Any) -> BoolRef:
        """Create the Z3 expression `other * self`.

        :param any other: The object to compare against.
        :return: The Z3 expression `other >= self`.
        :rtype: BoolRef
        """
        return self.z3.__truediv__(_coerce(other))

    def cast(self, val: Any) -> T:
        """Cast the supplied value to the datatype T of the variable.

        :param any val: The value to cast.
        :return: The supplied value as an instance of T.
        :rtype: T
        """
        assert val is not None, f"Invalid value None for variable {self}"
        if isinstance(val, RatNumRef) and self.datatype == float:
            return float(val.numerator().as_long() / val.denominator().as_long())
        if hasattr(val, "is_string_value") and val.is_string_value() and self.datatype == str:
            return val.as_string()
        if (isinstance(val, float) or isinstance(val, int)) and (self.datatype == int or self.datatype == float):
            return self.datatype(val)
        return self.datatype(str(val))

    def z3_val(self, z3_var, val: Any) -> T:
        native_val = self.cast(val)
        if isinstance(native_val, Enum):
            values = [z3_var.sort().constructor(c)() for c in range(z3_var.sort().num_constructors())]
            values = [v for v in values if str(v) == str(val)]
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

    @abstractmethod
    def copy(self, name: str = None) -> Variable:
        """Return a new instance of the Variable with the given name, or with
        the original name if no name is supplied.

        :param str name: The variable name.
        :return: A new Variable instance.
        :rtype: Variable

        """
        raise NotImplementedError("Method `copy` must be instantiated.")


class Input(Variable):
    """An extension of the Variable class representing inputs."""

    def copy(self, name=None) -> Input:
        if name:
            return Input(name, self.datatype, self.distribution)
        return Input(self.name, self.datatype, self.distribution)


class Output(Variable):
    """An extension of the Variable class representing outputs."""

    def copy(self, name=None) -> Output:
        if name:
            return Output(name, self.datatype, self.distribution)
        return Output(self.name, self.datatype, self.distribution)


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

    def copy(self, name=None) -> Meta:
        if name:
            return Meta(name, self.datatype, self.distribution)
        return Meta(self.name, self.datatype, self.distribution)
