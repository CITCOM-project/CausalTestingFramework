from abc import ABC, abstractmethod
from .causal_test_case import CausalTestResult

# TODO: Is it better to have these implemented this way or as a concrete class
# for which the user just specifies the function? We could then have "shortcut
# functions" to allow the user to quickly grab the common effects


class TestOutcome(ABC):
    @abstractmethod
    def apply(res: CausalTestResult) -> bool:
        pass


class Positive(TestOutcome):
    def apply(res: CausalTestResult) -> bool:
        # TODO: confidence intervals?
        return res.value > 0


class Negative(TestOutcome):
    def apply(res: CausalTestResult) -> bool:
        # TODO: confidence intervals?
        return res.value < 0


class NoEffect(TestOutcome):
    def apply(res: CausalTestResult) -> bool:
        return res.ci_low < 0 < res.ci_high
