from causal_testing.testing.intervention import Intervention
from causal_testing.testing.causal_test_outcome import CausalTestOutcome
from causal_testing.specification.scenario import Scenario
from causal_testing.specification.variable import Variable
import z3


class AbstractCausalTestCase:
    """
    An abstract test case serves as a generator for concrete test cases. Instead of having concrete conctrol
    and treatment values, we instead just specify the intervention and the treatment variables. This then
    enables potentially infinite concrete test cases to be generated between different values of the treatment.
    """

    def __init__(
        self,
        scenario_constraints: {z3.ExprRef},
        intervention: Intervention,
        treatment_vars: {Variable},
        expected_causal_effect: {Variable: ExprRef},
        effect_modifiers: {Variable} = {},
    ):
        self.scenario_constraints = scenario_constraints
        self.intervention = intervention
        self.treatment_vars = treatment_vars
        self.expected_causal_effect = expected_causal_effect
        self.effect_modifiers = effect_modifiers

    def generate_concrete_tests(num: int) -> [CausalTestCase]:
        """Generates a list of `num` concrete test cases.

        :param int num: The number of test cases to generate.
        :return: Description of returned object.
        :rtype: [CausalTestCase]

        """
        pass


class CausalTestCase:
    """
    A causal test case is a triple (X, Delta, Y), where X is an input configuration, Delta is an intervention, and
    Y is the expected causal effect on a particular output. The goal of a causal test case is to test whether the
    intervention Delta made to the input configuration X causes the model-under-test to produce the expected change
    in Y.
    """

    def __init__(
        self,
        input_configuration: {Variable: any},
        intervention: Intervention,
        expected_causal_effect: {Variable: CausalTestOutcome},
    ):
        """
        When a CausalTestCase is initialised, it takes the intervention and applies it to the input configuration to
        create two distinct input configurations: a control input configuration and a treatment input configuration.
        The former is the input configuration before applying the intervention and the latter is the input configuration
        after applying the intervention.

        :param {Variable: any} input_configuration: The input configuration representing the control values of the treatment variables.
        :param Intervention intervention: The metamorphic operator which transforms the control configuration to the treatment configuration.
        :param {Variable: CausalTestOutcome} The expected outcome.
        """
        self.control_input_configuration = input_configuration
        self.intervention = intervention
        self.treatment_input_configuration = intervention.apply(
            self.control_input_configuration
        )
        self.expected_causal_effect = expected_causal_effect

    def __str__(self):
        return (
            f"Applying {self.intervention} to {self.control_input_configuration} should cause the following "
            f"changes: {self.expected_causal_effect}."
        )


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


if __name__ == "__main__":
    test_results = CausalTestResult("y ~ x0*t1 + x1*z0", 100, [90, 110], 0.05)
    print(test_results)
