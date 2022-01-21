from abc import ABC, abstractmethod
from typing import Union


class CausalTestResult:
    """ A container to hold the results of a causal test case. Every causal test case provides a point estimate of
        the ATE, given a particular treatment, outcome, and adjustment set. Some but not all estimators can provide
        confidence intervals. """

    def __init__(self, treatment: tuple, outcome: tuple, treatment_value: Union[int, float, str],
                 control_value: Union[int, float, str], adjustment_set: float, ate: float,
                 confidence_intervals: [float, float] = None):
        self.treatment = treatment
        self.outcome = outcome
        self.treatment_value = treatment_value
        self.control_value = control_value
        if adjustment_set:
            self.adjustment_set = adjustment_set
        else:
            self.adjustment_set = '{}'
        self.ate = ate
        self.confidence_intervals = confidence_intervals

    def __str__(self):
        base_str = f"Causal Test Result\n==============\n" \
                   f"Treatment: {self.treatment[0]}\n" \
                   f"Control value: {self.control_value}\n" \
                   f"Treatment value: {self.treatment_value}\n" \
                   f"Outcome: {self.outcome[0]}\n" \
                   f"Adjustment set: {self.adjustment_set}\n" \
                   f"ATE: {self.ate}\n"
        confidence_str = ""
        if self.confidence_intervals:
            confidence_str += f"Confidence intervals: {self.confidence_intervals}\n"
        return base_str + confidence_str

    def ci_low(self):
        """Return the lower bracket of the confidence intervals."""
        return min(self.confidence_intervals)

    def ci_high(self):
        """Return the higher bracket of the confidence intervals."""
        return max(self.confidence_intervals)

    def summary(self):
        """Summarise the causal test result as an intuitive sentence."""
        print(f'The causal effect of changing {self.treatment[0]} = {self.control_value} to '
              f'{self.treatment[0]}\' = {self.treatment_value} is {self.ate} (95% confidence intervals: '
              f'{self.confidence_intervals}).')


class CausalTestOutcome(ABC):
    """An abstract class representing an expected causal effect."""

    @abstractmethod
    def apply(self, res: CausalTestResult) -> bool:
        pass

    def __str__(self) -> str:
        return type(self).__name__


class ExactValue(CausalTestOutcome):
    """An extension of TestOutcome representing that the expected causal effect should be a specific value."""
    def __init__(self, value: float):
        self.value = value

    def apply(self, res: CausalTestResult) -> bool:
        return res.ate == self.value


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
