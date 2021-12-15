from causal_testing.specification.scenario import Scenario
from causal_testing.specification.variable import Variable
from causal_testing.testing.causal_test_case import CausalTestCase
from causal_testing.testing.causal_test_outcome import CausalTestOutcome
import z3
import pandas as pd
import lhsmdu

import logging
logger = logging.getLogger(__name__)


class AbstractCausalTestCase:
    """
    An abstract test case serves as a generator for concrete test cases. Instead of having concrete conctrol
    and treatment values, we instead just specify the intervention and the treatment variables. This then
    enables potentially infinite concrete test cases to be generated between different values of the treatment.
    """

    def __init__(self, scenario: Scenario, intervention_constraints: {z3.ExprRef}, treatment_variables: {Variable},
                 expected_causal_effect: CausalTestOutcome, outcome_variables: {Variable},
                 effect_modifiers: {Variable} = None):
        assert treatment_variables.issubset(scenario.variables.values()), (
            "Treatment variables must be a subset of variables."
            + " Instead got:\ntreatment_variables={treatment_variables}\nvariables={variables}"
        )

        self.scenario = scenario
        self.intervention_constraints = intervention_constraints
        self.treatment_variables = treatment_variables
        self.expected_causal_effect = expected_causal_effect
        self.outcome_variables = outcome_variables

        if effect_modifiers is not None:
            self.effect_modifiers = effect_modifiers
        else:
            self.effect_modifiers = {}

    def generate_concrete_tests(self, sample_size: int, ) -> ([CausalTestCase], pd.DataFrame):
        """Generates a list of `num` concrete test cases.

        :param sample_size: The number of test cases to generate.
        :return: A list of causal test cases and a dataframe representing the required model run configurations.
        """
        # Generate the Latin Hypercube samples and put into a dataframe
        samples = pd.DataFrame(
            lhsmdu.sample(len(self.scenario.inputs()), sample_size).T,
            columns=[v.name for v in self.scenario.inputs()],
        )
        # Project the samples to the variables' distributions
        for var in self.scenario.inputs():
            # TODO: This only works for Inputs. We need to do it for Metas too...
            samples[var.name] = lhsmdu.inverseTransformSample(
                var.distribution, samples[var.name]
            )

        concrete_tests = []
        runs = []
        run_columns = sorted([v.name for v in self.scenario.inputs()])
        for _, row in samples.iterrows():
            optimizer = z3.Optimize()
            for i, c in enumerate(self.scenario.constraints):
                optimizer.assert_and_track(c, f"constraint_{i}")
            for i, c in enumerate(self.intervention_constraints):
                optimizer.assert_and_track(c, f"intervention_{i}")

            optimizer.add_soft([v.z3 == row[v.name] for v in self.scenario.inputs()])
            sat = optimizer.check()
            if sat == z3.unsat:
                logger.warning(
                    "Satisfiability of test case was unsat.\n"
                    + f"Constraints\n{optimizer}\nUnsat core {optimizer.unsat_core()}"
                )
            model = optimizer.model()
            concrete_test = CausalTestCase(
                control_input_configuration={v: v.cast(model[v.z3]) for v in self.treatment_variables},
                expected_causal_effect=self.expected_causal_effect,
                outcome_variables=self.outcome_variables,
                treatment_input_configuration={
                    v: v.cast(model[self.scenario.treatment_variables[v.name].z3])
                    for v in self.treatment_variables
                },
            )
            concrete_tests.append(concrete_test)
            runs.append(
                {
                    v.name: v.cast(model[v.z3])
                    for v in self.scenario.variables.values()
                    if v.name in run_columns
                }
            )
        return concrete_tests, pd.DataFrame(runs, columns=run_columns)
