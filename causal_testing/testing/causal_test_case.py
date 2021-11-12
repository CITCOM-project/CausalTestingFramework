from abc import ABC, abstractmethod
from intervention import Intervention


class CausalTestCase(ABC):
    """
    A causal test case is a triple (X, Delta, Y), where X is an input configuration, Delta is an intervention, and
    Y is the expected causal effect on a particular output. The goal of a causal test case is to test whether the
    intervention Delta made to the input configuration X causes the model-under-test to produce the expected change
    in Y. To this end, a causal test case will use causal inference and causal knowledge (from the causal specification)
    to design a statistical experiment which can isolate the causal effect of interest.
    """

    def __init__(self, input_configuration: dict, intervention: Intervention, expected_causal_effect: float):
        self.input_configuration = input_configuration
        self.intervention = intervention
        self.expected_causal_effect = expected_causal_effect
        super().__init__()

    @abstractmethod
    def apply_intervention(self):
        """
        A causal test case should apply the intervention to the input configuration, creating a treated input
        configuration.
        """
        pass

    @abstractmethod
    def collect_data(self):
        """
        Collect data for causal testing. The user can do this in two ways: (1) experimental - run the model to
        gather the data (running an RCT); (2) observational - the user provides existing data from previous model
        runs. In the observational case, the data should be filtered to ensure it is valid for the given modelling
        scenario.
        """

    @abstractmethod
    def estimate_causal_effect(self):
        """
        A causal test case should estimate the causal effect of the intervention on the outcome of interest.
        """


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


if __name__ == "__main__":
    test_results = CausalTestResult("y ~ x0*t1 + x1*z0", 100, [90, 110], 0.05)
    print(test_results)
