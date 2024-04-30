"""This module contains the class AbstractCausalTestCase, which generates concrete test cases"""

import itertools
import logging
from enum import Enum
from typing import Iterable

import lhsmdu
import pandas as pd
import z3
from scipy import stats


from causal_testing.specification.scenario import Scenario
from causal_testing.specification.variable import Variable
from causal_testing.testing.causal_test_case import CausalTestCase
from causal_testing.testing.causal_test_outcome import CausalTestOutcome
from causal_testing.testing.base_test_case import BaseTestCase


logger = logging.getLogger(__name__)


class AbstractCausalTestCase:
    """
    An abstract test case serves as a generator for concrete test cases. Instead of having concrete control
    and treatment values, we instead just specify the intervention and the treatment variables. This then
    enables potentially infinite concrete test cases to be generated between different values of the treatment.
    """

    def __init__(
        # pylint: disable=too-many-arguments
        self,
        scenario: Scenario,
        intervention_constraints: set[z3.ExprRef],
        treatment_variable: Variable,
        expected_causal_effect: dict[Variable:CausalTestOutcome],
        effect_modifiers: set[Variable] = None,
        estimate_type: str = "ate",
        effect: str = "total",
    ):
        if treatment_variable not in scenario.variables.values():
            raise ValueError(
                "Treatment variables must be a subset of variables."
                + f" Instead got:\ntreatment_variables={treatment_variable}\nvariables={scenario.variables}"
            )

        assert len(expected_causal_effect) == 1, "We currently only support tests with one causal outcome"

        self.scenario = scenario
        self.intervention_constraints = intervention_constraints
        self.treatment_variable = treatment_variable
        self.expected_causal_effect = expected_causal_effect
        self.estimate_type = estimate_type
        self.effect = effect

        if effect_modifiers is not None:
            self.effect_modifiers = effect_modifiers
        else:
            self.effect_modifiers = {}

    def __str__(self):
        outcome_string = " and ".join(
            [f"the effect on {var} should be {str(effect)}" for var, effect in self.expected_causal_effect.items()]
        )
        return f"When we apply intervention {self.intervention_constraints}, {outcome_string}"

    def datapath(self) -> str:
        """Create and return the sanitised data path"""

        def sanitise(string):
            return "".join([x for x in string if x.isalnum()])

        return (
            sanitise("-".join([str(c) for c in self.intervention_constraints]))
            + "_"
            + "-".join([f"{v.name}_{e}" for v, e in self.expected_causal_effect.items()])
            + ".csv"
        )

    def _generate_concrete_tests(
        # pylint: disable=too-many-locals
        self,
        sample_size: int,
        rct: bool = False,
        seed: int = 0,
    ) -> tuple[list[CausalTestCase], pd.DataFrame]:
        """Generates a list of `num` concrete test cases.

        :param sample_size: The number of strata to use for Latin hypercube sampling. Where no target_ks_score is
        provided, this corresponds to the number of test cases to generate. Where target_ks_score is provided, the
        number of test cases will be a multiple of this.
        :param rct: Whether we're running an RCT, i.e. whether to add the treatment run to the concrete runs.
        :param seed: Random seed for reproducability.
        :return: A list of causal test cases and a dataframe representing the required model run configurations.
        :rtype: ([CausalTestCase], pd.DataFrame)
        """

        concrete_tests = []
        runs = []
        run_columns = sorted([v.name for v in self.scenario.variables.values() if v.distribution])

        # Generate the Latin Hypercube samples and put into a dataframe
        # lhsmdu.setRandomSeed(seed+i)
        samples = pd.DataFrame(
            lhsmdu.sample(len(run_columns), sample_size, randomSeed=seed).T,
            columns=run_columns,
        )
        # Project the samples to the variables' distributions
        for name in run_columns:
            var = self.scenario.variables[name]
            samples[var.name] = lhsmdu.inverseTransformSample(var.distribution, samples[var.name])

        for index, row in samples.iterrows():
            model = self._optimizer_model(run_columns, row)

            base_test_case = BaseTestCase(
                treatment_variable=self.treatment_variable,
                outcome_variable=list(self.expected_causal_effect.keys())[0],
                effect=self.effect,
            )

            concrete_test = CausalTestCase(
                base_test_case=base_test_case,
                control_value=self.treatment_variable.cast(model[self.treatment_variable.z3]),
                treatment_value=self.treatment_variable.cast(
                    model[self.scenario.treatment_variables[self.treatment_variable.name].z3]
                ),
                expected_causal_effect=list(self.expected_causal_effect.values())[0],
                estimate_type=self.estimate_type,
                effect_modifier_configuration={v: v.cast(model[v.z3]) for v in self.effect_modifiers},
            )

            for v in self.scenario.inputs():
                if v.name in row and row[v.name] != v.cast(model[v.z3]):
                    constraints = "\n  ".join([str(c) for c in self.scenario.constraints if v.name in str(c)])
                    logger.warning(
                        f"Unable to set variable {v.name} to {row[v.name]} because of constraints\n"
                        + f"{constraints}\nUsing value {v.cast(model[v.z3])} instead in test\n{concrete_test}"
                    )

            if not any((vars(t) == vars(concrete_test) for t in concrete_tests)):
                concrete_tests.append(concrete_test)
                # Control run
                control_run = {
                    v.name: v.cast(model[v.z3]) for v in self.scenario.variables.values() if v.name in run_columns
                }
                control_run["bin"] = index
                runs.append(control_run)
                # Treatment run
                if rct:
                    treatment_run = control_run.copy()
                    treatment_run.update({concrete_test.treatment_variable.name: concrete_test.treatment_value})
                    treatment_run["bin"] = index
                    runs.append(treatment_run)

        return concrete_tests, pd.DataFrame(runs, columns=run_columns + ["bin"])

    def generate_concrete_tests(
        # pylint: disable=too-many-arguments, too-many-locals
        self,
        sample_size: int,
        target_ks_score: float = None,
        rct: bool = False,
        seed: int = 0,
        hard_max: int = 1000,
    ) -> tuple[list[CausalTestCase], pd.DataFrame]:
        """Generates a list of `num` concrete test cases.

        :param sample_size: The number of strata to use for Latin hypercube sampling. Where no target_ks_score is
        provided, this corresponds to the number of test cases to generate. Where target_ks_score is provided, the
        number of test cases will be a multiple of this.
        :param target_ks_score: The target KS score. A value in range [0, 1] with lower values representing a higher
        confidence and requireing more tests to achieve. A value of 0.05 is recommended.
        TODO: Make this more flexible so we're not restricting ourselves just to the KS test.
        :param rct: Whether we're running an RCT, i.e. whether to add the treatment run to the concrete runs.
        :param seed: Random seed for reproducability.
        :param hard_max: Number of iterations to run for before timing out if target_ks_score cannot be reached.
        :return: A list of causal test cases and a dataframe representing the required model run configurations.
        :rtype: ([CausalTestCase], pd.DataFrame)
        """

        if target_ks_score is not None:
            assert 0 <= target_ks_score <= 1, "target_ks_score must be between 0 and 1."
        else:
            hard_max = 1

        concrete_tests = []
        runs = pd.DataFrame()
        ks_stats = []

        pre_break = False
        for i in range(hard_max):
            concrete_tests_temp, runs_temp = self._generate_concrete_tests(sample_size, rct, seed + i)
            for test in concrete_tests_temp:
                if not any((vars(test) == vars(t) for t in concrete_tests)):
                    concrete_tests.append(test)
            runs = pd.concat([runs, runs_temp])
            assert concrete_tests_temp not in concrete_tests, "Duplicate entries unlikely unless something went wrong"

            control_configs = pd.DataFrame([{test.treatment_variable: test.control_value} for test in concrete_tests])
            ks_stats = {
                var: stats.kstest(control_configs[var], var.distribution.cdf).statistic
                for var in control_configs.columns
            }
            # Putting treatment and control values in messes it up because the two are not independent...
            # This is potentially problematic as constraints might mean we don't get good coverage if we use control
            # values alone
            # We might then need to carefully craft our _control value_ generating distributions so that we can get
            # good coverage
            # without the generated treatment values violating any constraints.

            # treatment_configs = pd.DataFrame([test.treatment_input_configuration for test in concrete_tests])
            # both_configs = pd.concat([control_configs, treatment_configs])
            # ks_stats = {var: stats.kstest(both_configs[var], var.distribution.cdf).statistic for var in
            # both_configs.columns}
            effect_modifier_configs = pd.DataFrame([test.effect_modifier_configuration for test in concrete_tests])
            ks_stats.update(
                {
                    var: stats.kstest(effect_modifier_configs[var], var.distribution.cdf).statistic
                    for var in effect_modifier_configs.columns
                }
            )
            control_values = [test.control_value for test in concrete_tests]
            treatment_values = [test.treatment_value for test in concrete_tests]

            if self.treatment_variable.datatype is bool and {(True, False), (False, True)}.issubset(
                set(zip(control_values, treatment_values))
            ):
                pre_break = True
                break
            if issubclass(self.treatment_variable.datatype, Enum) and set(
                {
                    (x, y)
                    for x, y in itertools.product(self.treatment_variable.datatype, self.treatment_variable.datatype)
                    if x != y
                }
            ).issubset(zip(control_values, treatment_values)):
                pre_break = True
                break
            if target_ks_score and all((stat <= target_ks_score for stat in ks_stats.values())):
                pre_break = True
                break

        if target_ks_score is not None and not pre_break:
            logger.error(
                "Hard max reached but could not achieve target ks_score of %s. Got %s. Generated %s distinct tests",
                target_ks_score,
                ks_stats,
                len(concrete_tests),
            )
        return concrete_tests, runs

    def _optimizer_model(self, run_columns: Iterable[str], row: pd.core.series) -> z3.Optimize:
        """
        :param run_columns: A sorted list of Variable names from the scenario variables
        :param row: A pandas Series containing a row from the Samples dataframe
        :return: z3 optimize model with constraints tracked and soft constraints added
        :rtype: z3.Optimize
        """
        optimizer = z3.Optimize()
        for c in self.scenario.constraints:
            optimizer.assert_and_track(c, str(c))
        for c in self.intervention_constraints:
            optimizer.assert_and_track(c, str(c))

        for v in run_columns:
            optimizer.add_soft(
                self.scenario.variables[v].z3
                == self.scenario.variables[v].z3_val(self.scenario.variables[v].z3, row[v])
            )

        if optimizer.check() == z3.unsat:
            logger.warning(
                f"Satisfiability of test case was unsat.\n"
                f"Constraints \n {optimizer} \n Unsat core {optimizer.unsat_core()}",
            )
        model = optimizer.model()
        return model
