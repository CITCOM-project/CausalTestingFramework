from causal_testing.testing.intervention import Intervention
from causal_testing.testing.causal_test_outcome import CausalTestOutcome
from causal_testing.specification.variable import Variable, Input
import z3
import lhsmdu
import pandas as pd

import logging

logger = logging.getLogger(__name__)


class CausalTestCase:
    """
    A causal test case is a triple (X, Delta, Y), where X is an input configuration, Delta is an intervention, and
    Y is the expected causal effect on a particular output. The goal of a causal test case is to test whether the
    intervention Delta made to the input configuration X causes the model-under-test to produce the expected change
    in Y.
    """

    def __init__(self, control_input_configuration: {Variable: any}, expected_causal_effect: CausalTestOutcome,
                 intervention: Intervention = None, treatment_input_configuration: {Variable: any} = None):
        """
        When a CausalTestCase is initialised, it takes the intervention and applies it to the input configuration to
        create two distinct input configurations: a control input configuration and a treatment input configuration.
        The former is the input configuration before applying the intervention and the latter is the input configuration
        after applying the intervention.

        :param {Variable: any} control_input_configuration: The input configuration representing the control values of
        the treatment variables.
        :param CausalTestOutcome The expected outcome.
        :param Intervention intervention: The metamorphic operator which transforms the control configuration to the
        treatment configuration. Defaults to None.
        :param {Variable: any} treatment_input_configuration: The input configuration representing the treatment
        values of the treatment variables.
        """
        assert (
            intervention is None or treatment_input_configuration is None
        ), "Cannot define both treatment configuration and intervention."

        self.control_input_configuration = control_input_configuration
        self.expected_causal_effect = expected_causal_effect
        self.intervention = intervention
        self.treatment_input_configuration = treatment_input_configuration
        if intervention:
            self.treatment_input_configuration = intervention.apply(
                self.control_input_configuration
            )

    def __str__(self):
        return (
            f"Applying {self.intervention} to {self.control_input_configuration} should cause the following "
            f"changes: {self.expected_causal_effect}."
        )

    def execute(data):
        raise NotImplementedError("Need to implement execute.")


class AbstractCausalTestCase:
    """
    An abstract test case serves as a generator for concrete test cases. Instead of having concrete conctrol
    and treatment values, we instead just specify the intervention and the treatment variables. This then
    enables potentially infinite concrete test cases to be generated between different values of the treatment.
    """

    def __init__(self, variables: {Variable}, primed_variables: {Variable}, scenario_constraints: {z3.ExprRef},
                 intervention_constraints: {z3.ExprRef}, treatment_variables: {Variable},
                 expected_causal_effect: {Variable: z3.ExprRef}, effect_modifiers: {Variable} = None):
        assert treatment_variables.issubset(variables), (
            "Treatment variables must be a subset of variables."
            + " Instead got:\ntreatment_variables={treatment_variables}\nvariables={variables}"
        )
        self.variables = variables
        self.primed_variables = primed_variables
        self.scenario_constraints = scenario_constraints
        self.intervention_constraints = intervention_constraints
        self.treatment_variables = treatment_variables
        self.expected_causal_effect = expected_causal_effect

        if effect_modifiers is not None:
            self.effect_modifiers = effect_modifiers
        else:
            self.effect_modifiers = {}

    def generate_concrete_tests(self, sample_size: int, ) -> ([CausalTestCase], pd.DataFrame):
        """Generates a list of `num` concrete test cases.

        :param int sample_size: The number of test cases to generate.
        :return: A list of causal test cases and a dataframe representing the required model run configurations.
        :rtype: ([CausalTestCase], pd.DataFrame)

        """
        # Generate the Latin Hypercube samples and put into a dataframe
        samples = pd.DataFrame(
            lhsmdu.sample(len(self.treatment_variables), sample_size).T,
            columns=[v.name for v in self.treatment_variables],
        )
        # Project the samples to the variables' distributions
        for var in self.treatment_variables:
            # TODO: This only works for Inputs. We need to do it for Metas too...
            samples[var.name] = lhsmdu.inverseTransformSample(
                var.distribution, samples[var.name]
            )

        concrete_tests = []
        runs = []
        run_columns = sorted([v.name for v in self.variables if isinstance(v, Input)])
        for _, row in samples.iterrows():
            optimizer = z3.Optimize()
            for i, c in enumerate(self.scenario_constraints):
                optimizer.assert_and_track(c, f"constraint_{i}")
            for i, c in enumerate(self.intervention_constraints):
                optimizer.assert_and_track(c, f"intervention_{i}")

            # optimizer.add(self.scenario_constraints)
            # optimizer.add(self.intervention_constraints)
            optimizer.add_soft([v.z3 == row[v.name] for v in self.treatment_variables])
            sat = optimizer.check()
            if sat == z3.unsat:
                logger.warn(
                    "Satisfiability of test case was unsat.\n"
                    + f"Constraints\n{optimizer}\nUnsat core {optimizer.unsat_core()}"
                )
            model = optimizer.model()
            concrete_tests.append(
                CausalTestCase(
                    {v: v.cast(model[v.z3]) for v in self.treatment_variables},
                    self.expected_causal_effect,
                    treatment_input_configuration={
                        v: v.cast(model[v.z3])
                        for v in self.primed_variables
                        if v.name in [v.name for v in self.treatment_variables]
                    },
                )
            )
            runs.append(
                {
                    v.name: v.cast(model[v.z3])
                    for v in self.variables
                    if v.name in run_columns
                }
            )
        return (concrete_tests, pd.DataFrame(runs, columns=run_columns))


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

    def apply_test_oracle_procedure(self, expected_causal_effect, *args, **kwargs) -> bool:
        """ Based on the results of the causal test case, determine whether the test passes or fails. By default, we
            check whether the casual estimate is equal to the expected causal effect. However, a user may override
            this method to define precise oracles. """
        # TODO: Work out the best way to implement test oracle procedure. A test oracle object?
        return self.ate == expected_causal_effect

