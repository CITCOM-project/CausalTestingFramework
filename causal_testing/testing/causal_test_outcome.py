from abc import ABC, abstractmethod

# TODO: Is it better to have these implemented this way or as a concrete class
# for which the user just specifies the function? We could then have "shortcut
# functions" to allow the user to quickly grab the common effects


class CausalTestResult:
    """ A container to hold the results of a causal test case. Every causal test case provides a point estimate of
        the ATE, given a particular treatment, outcome, and adjustment set. Some but not all estimators can provide
        confidence intervals. """

    def __init__(self, adjustment_set: float, ate: float, confidence_intervals: [float, float] = None,
                 confidence_level: float = None):
        self.adjustment_set = adjustment_set
        self.ate = ate
        self.confidence_intervals = confidence_intervals
        self.confidence_level = confidence_level

    def __str__(self):
        base_str = f"Adjustment set: {self.adjustment_set}\nATE: {self.ate}\n"
        confidence_str = ""
        if self.confidence_intervals:
            confidence_str += f"Confidence intervals: {self.confidence_intervals}\n"
        if self.confidence_level:
            confidence_str += f"Confidence level: {self.confidence_level}"
        return base_str + confidence_str

    def ci_low(self):
        return min(self.confidence_intervals)

    def ci_high(self):
        return max(self.confidence_intervals)

    # def apply_test_oracle_procedure(self, expected_causal_effect, *args, **kwargs) -> bool:
    #     """ Based on the results of the causal test case, determine whether the test passes or fails. By default, we
    #         check whether the casual estimate is equal to the expected causal effect. However, a user may override
    #         this method to define precise oracles. """
    #     # TODO: Work out the best way to implement test oracle procedure. A test oracle object?
    #     return self.ate == expected_causal_effect


class CausalTestOutcome(ABC):
    """An abstract class representing an expected causal effect."""

    @abstractmethod
    def apply(self, res: CausalTestResult) -> bool:
        pass


class Positive(CausalTestOutcome):
    """An extension of TestOutcome representing that the expected causal effect should be positive."""

    def apply(self, res: CausalTestResult) -> bool:
        # TODO: confidence intervals?
        return res.ate > 0


class Negative(CausalTestOutcome):
    """An extension of TestOutcome representing that the expected causal effect should be negative."""

    def apply(self, res: CausalTestResult) -> bool:
        # TODO: confidence intervals?
        return res.ate < 0


class NoEffect(CausalTestOutcome):
    """An extension of TestOutcome representing that the expected causal effect should be zero."""

    def apply(self, res: CausalTestResult) -> bool:
        return res.ci_low() < 0 < res.ci_high()
