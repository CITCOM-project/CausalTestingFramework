import unittest
import os
import pandas as pd
import numpy as np
from causal_testing.generation.abstract_causal_test_case import AbstractCausalTestCase
from causal_testing.generation.enum_gen import EnumGen
from causal_testing.specification.causal_specification import Scenario
from causal_testing.specification.variable import Input, Output
from scipy.stats import uniform, rv_discrete
from tests.test_helpers import create_temp_dir_if_non_existent, remove_temp_dir_if_existent
from causal_testing.testing.causal_test_outcome import Positive
from z3 import And
from enum import Enum


class Car(Enum):
    isetta = "vehicle.bmw.isetta"
    mkz2017 = "vehicle.lincoln.mkz2017"

    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented


class TestAbstractTestCase(unittest.TestCase):
    """
    Class to test abstract test cases.
    """

    def setUp(self) -> None:
        temp_dir_path = create_temp_dir_if_non_existent()
        self.dag_dot_path = os.path.join(temp_dir_path, "dag.dot")
        self.observational_df_path = os.path.join(temp_dir_path, "observational_data.csv")
        # Y = 3*X1 + X2*X3 + 10
        self.observational_df = pd.DataFrame({"X1": [1, 2, 3, 4], "X2": [5, 6, 7, 8], "X3": [10, 20, 30, 40]})
        self.observational_df["Y"] = self.observational_df.apply(
            lambda row: (3 * row.X1) + (row.X2 * row.X3) + 10, axis=1
        )
        self.observational_df.to_csv(self.observational_df_path)
        self.X1 = Input("X1", float, uniform(1, 4))
        self.X2 = Input("X2", int, rv_discrete(values=([7], [1])))
        self.X3 = Input("X3", float, uniform(10, 40))
        self.X4 = Input("X4", int, rv_discrete(values=([10], [1])))
        self.X5 = Input("X5", bool, rv_discrete(values=(range(2), [0.5, 0.5])))
        self.Car = Input("Car", Car, EnumGen(Car))
        self.Y = Output("Y", int)

    def test_generate_concrete_test_cases(self):
        scenario = Scenario({self.X1, self.X2, self.X3, self.X4})
        scenario.setup_treatment_variables()
        abstract = AbstractCausalTestCase(
            scenario=scenario,
            intervention_constraints={scenario.treatment_variables[self.X1.name].z3 > self.X1.z3},
            treatment_variable=self.X1,
            expected_causal_effect={self.Y: Positive()},
            effect_modifiers=None,
        )
        concrete_tests, runs = abstract.generate_concrete_tests(2)
        assert len(concrete_tests) == 2, "Expected 2 concrete tests"
        assert len(runs) == 2, "Expected 2 runs"

    def test_generate_boolean_concrete_test_cases(self):
        scenario = Scenario({self.X1, self.X2, self.X3, self.X5})
        scenario.setup_treatment_variables()
        abstract = AbstractCausalTestCase(
            scenario=scenario,
            intervention_constraints={
                scenario.treatment_variables[self.X5.name].z3 != scenario.variables[self.X5.name].z3
            },
            treatment_variable=self.X5,
            expected_causal_effect={self.Y: Positive()},
            effect_modifiers=None,
        )
        concrete_tests, runs = abstract.generate_concrete_tests(2)
        assert len(concrete_tests) == 2, "Expected 2 concrete test"
        assert len(runs) == 2, "Expected 2 run"

    def test_generate_enum_concrete_test_cases(self):
        scenario = Scenario({self.Car})
        scenario.setup_treatment_variables()
        abstract = AbstractCausalTestCase(
            scenario=scenario,
            intervention_constraints={
                scenario.treatment_variables[self.Car.name].z3 != scenario.variables[self.Car.name].z3
            },
            treatment_variable=self.Car,
            expected_causal_effect={self.Y: Positive()},
            effect_modifiers=None,
        )
        concrete_tests, runs = abstract.generate_concrete_tests(10)
        assert len(concrete_tests) == 2, "Expected 2 concrete tests"
        assert len(runs) == 2, "Expected 2 runs"

    def test_str(self):
        scenario = Scenario({self.X1, self.X2, self.X3, self.X4})
        scenario.setup_treatment_variables()
        abstract = AbstractCausalTestCase(
            scenario=scenario,
            intervention_constraints={scenario.treatment_variables[self.X1.name].z3 > self.X1.z3},
            treatment_variable=self.X1,
            expected_causal_effect={self.Y: Positive()},
            effect_modifiers=None,
        )
        assert (
            str(abstract) == "When we apply intervention {X1' > X1}, the effect on Output: Y::int should be Positive"
        ), f"Unexpected string {str(abstract)}"

    def test_datapath(self):
        scenario = Scenario({self.X1, self.X2, self.X3, self.X4})
        scenario.setup_treatment_variables()
        abstract = AbstractCausalTestCase(
            scenario=scenario,
            intervention_constraints={scenario.treatment_variables[self.X1.name].z3 > self.X1.z3},
            treatment_variable=self.X1,
            expected_causal_effect={self.Y: Positive()},
            effect_modifiers=None,
        )
        assert abstract.datapath() == "X1X1_Y_Positive.csv", f"Unexpected datapath {abstract.datapath()}"

    def test_generate_concrete_test_cases_with_constraints(self):
        scenario = Scenario({self.X1, self.X2, self.X3, self.X4}, {self.X1 < self.X2})
        scenario.setup_treatment_variables()
        abstract = AbstractCausalTestCase(
            scenario=scenario,
            intervention_constraints={scenario.treatment_variables[self.X1.name].z3 > self.X1.z3},
            treatment_variable=self.X1,
            expected_causal_effect={self.Y: Positive()},
            effect_modifiers=None,
        )
        concrete_tests, runs = abstract.generate_concrete_tests(2)
        assert len(concrete_tests) == 2, "Expected 2 concrete tests"
        assert len(runs) == 2, "Expected 2 runs"

    def test_generate_concrete_test_cases_with_effect_modifiers(self):
        scenario = Scenario({self.X1, self.X2, self.X3, self.X4})
        scenario.setup_treatment_variables()
        abstract = AbstractCausalTestCase(
            scenario=scenario,
            intervention_constraints={scenario.treatment_variables[self.X1.name].z3 > self.X1.z3},
            treatment_variable=self.X1,
            expected_causal_effect={self.Y: Positive()},
            effect_modifiers={self.X2},
        )
        concrete_tests, runs = abstract.generate_concrete_tests(2)
        assert len(concrete_tests) == 2, "Expected 2 concrete tests"
        assert len(runs) == 2, "Expected 2 runs"

    def test_generate_concrete_test_cases_rct(self):
        scenario = Scenario({self.X1, self.X2, self.X3, self.X4})
        scenario.setup_treatment_variables()
        abstract = AbstractCausalTestCase(
            scenario=scenario,
            intervention_constraints={scenario.treatment_variables[self.X1.name].z3 > self.X1.z3},
            treatment_variable=self.X1,
            expected_causal_effect={self.Y: Positive()},
            effect_modifiers=None,
        )
        concrete_tests, runs = abstract.generate_concrete_tests(2, rct=True)
        assert len(concrete_tests) == 2, "Expected 2 concrete tests"
        assert len(runs) == 4, "Expected 4 runs"

    def test_infeasible_constraints(self):
        scenario = Scenario({self.X1, self.X2, self.X3, self.X4}, [self.X1.z3 > 2])
        scenario.setup_treatment_variables()
        abstract = AbstractCausalTestCase(
            scenario=scenario,
            intervention_constraints={scenario.treatment_variables[self.X1.name].z3 > self.X1.z3},
            treatment_variable=self.X1,
            expected_causal_effect={self.Y: Positive()},
            effect_modifiers=None,
        )
        HARD_MAX = 10
        NUM_STRATA = 4

        with self.assertWarns(Warning):
            concrete_tests, runs = abstract.generate_concrete_tests(4, rct=True, target_ks_score=0.1, hard_max=HARD_MAX)
        self.assertTrue(all((x > 2 for x in runs["X1"])))
        self.assertTrue(len(concrete_tests) <= HARD_MAX * NUM_STRATA)

    def test_feasible_constraints(self):
        scenario = Scenario({self.X1, self.X2, self.X3, self.X4})
        scenario.setup_treatment_variables()
        abstract = AbstractCausalTestCase(
            scenario=scenario,
            intervention_constraints={scenario.treatment_variables[self.X1.name].z3 > self.X1.z3},
            treatment_variable=self.X1,
            expected_causal_effect={self.Y: Positive()},
            effect_modifiers=None,
        )
        concrete_tests, _ = abstract.generate_concrete_tests(4, rct=True, target_ks_score=0.1, hard_max=1000)
        assert len(concrete_tests) < 1000

    def tearDown(self) -> None:
        remove_temp_dir_if_existent()


if __name__ == "__main__":
    unittest.main()
