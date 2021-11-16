from abc import ABC
from intervention import Intervention


class CausalTestCase(ABC):
    """
    A causal test case is a triple (X, Delta, Y), where X is an input configuration, Delta is an intervention, and
    Y is the expected causal effect on a particular output. The goal of a causal test case is to test whether the
    intervention Delta made to the input configuration X causes the model-under-test to produce the expected change
    in Y.
    """

    def __init__(self, input_configuration: dict, intervention: Intervention, expected_causal_effect: {str: float}):
        self.control_input_configuration = input_configuration
        self.intervention = intervention
        self.treatment_input_configuration = intervention.apply(self.control_input_configuration)
        self.expected_causal_effect = expected_causal_effect

    def __str__(self):
        return f'Applying {self.intervention} to {self.control_input_configuration} should cause the following ' \
               f'changes: {self.expected_causal_effect}.'


class CausalTestResult(ABC):
    """ A container to hold the results of a causal test case. Every causal test case provides a point estimate of
        the ATE for a particular estimand. Some but not all estimators can provide confidence intervals. """

    def __init__(self, estimand: float, point_estimate: float, confidence_intervals: [float, float] = None,
                 confidence_level: float = None):
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

    def apply_test_oracle_procedure(self):
        """ Based on the results of the causal test case, determine whether the test passes or fails. """
        pass


if __name__ == "__main__":
    test_results = CausalTestResult("y ~ x0*t1 + x1*z0", 100, [90, 110], 0.05)
    print(test_results)
