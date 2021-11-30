from abc import ABC, abstractmethod
from .causal_test_case import CausalTestResult

# TODO: Is it better to have these implemented this way or as a concrete class
# for which the user just specifies the function? We could then have "shortcut
# functions" to allow the user to quickly grab the common effects


class TestOutcome(ABC):
    """An abstract class representing an expected causal effect."""

    @abstractmethod
    def apply(res: CausalTestResult) -> bool:
        pass


class Positive(TestOutcome):
    """An extension of TestOutcome representing that the expected causal effect should be positive."""

    def apply(res: CausalTestResult) -> bool:
        # TODO: confidence intervals?
        return res.value > 0


class Negative(TestOutcome):
    """An extension of TestOutcome representing that the expected causal effect should be negative."""

    def apply(res: CausalTestResult) -> bool:
        # TODO: confidence intervals?
        return res.value < 0


class NoEffect(TestOutcome):
    """An extension of TestOutcome representing that the expected causal effect should be zero."""

    def apply(res: CausalTestResult) -> bool:
        return res.ci_low < 0 < res.ci_high
