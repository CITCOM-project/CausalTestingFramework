from abc import ABC, abstractmethod

# TODO: Is it better to have these implemented this way or as a concrete class
# for which the user just specifies the function? We could then have "shortcut
# functions" to allow the user to quickly grab the common effects


class CausalTestResult:
    """ A container to hold the results of a causal test case. Every causal test case provides a point estimate of
        the ATE for a particular estimand. Some but not all estimators can provide confidence intervals. """

    def __init__(
        self,
        estimand: float,
        point_estimate: float,
        confidence_intervals: [float, float] = None,
        confidence_level: float = None,
    ):
        self.estimand = estimand
        self.point_estimate = point_estimate
        self.confidence_intervals = confidence_intervals
        self.confidence_level = confidence_level

    def __str__(self):
        base_str = f"Estimand: {self.estimand}\nATE: {self.point_estimate}\n"
        confidence_str = ""
        if self.confidence_intervals:
            confidence_str += f"Confidence intervals: {self.confidence_intervals}\n"
        if self.confidence_level:
            confidence_str += f"Confidence level: {self.confidence_level}"
        return base_str + confidence_str

    def apply_test_oracle_procedure(
        self, expected_causal_effect, *args, **kwargs
    ) -> bool:
        """ Based on the results of the causal test case, determine whether the test passes or fails. By default, we
            check whether the casual estimate is equal to the expected causal effect. However, a user may override
            this method to define precise oracles. """
        # TODO: Work out the best way to implement test oracle procedure. A test oracle object?
        return self.point_estimate == expected_causal_effect


class CausalTestOutcome(ABC):
    """An abstract class representing an expected causal effect."""

    @abstractmethod
    def apply(res: CausalTestResult) -> bool:
        pass


class Positive(CausalTestOutcome):
    """An extension of TestOutcome representing that the expected causal effect should be positive."""

    def apply(res: CausalTestResult) -> bool:
        # TODO: confidence intervals?
        return res.value > 0


class Negative(CausalTestOutcome):
    """An extension of TestOutcome representing that the expected causal effect should be negative."""

    def apply(res: CausalTestResult) -> bool:
        # TODO: confidence intervals?
        return res.value < 0


class NoEffect(CausalTestOutcome):
    """An extension of TestOutcome representing that the expected causal effect should be zero."""

    def apply(res: CausalTestResult) -> bool:
        return res.ci_low < 0 < res.ci_high
