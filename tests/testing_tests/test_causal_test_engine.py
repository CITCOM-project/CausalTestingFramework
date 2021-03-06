import unittest
import os
import pandas as pd
import numpy as np
from tests.test_helpers import create_temp_dir_if_non_existent, remove_temp_dir_if_existent
from causal_testing.specification.causal_specification import CausalSpecification, Scenario
from causal_testing.specification.variable import Input, Output
from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.data_collection.data_collector import ObservationalDataCollector
from causal_testing.testing.causal_test_case import CausalTestCase
from causal_testing.testing.causal_test_engine import CausalTestEngine
from causal_testing.testing.causal_test_outcome import ExactValue
from causal_testing.testing.estimators import CausalForestEstimator, LinearRegressionEstimator


class TestCausalTestEngineObservational(unittest.TestCase):
    """ Test the CausalTestEngine workflow using observational data.

    The causal test engine (CTE) is the main workflow for the causal testing framework. The CTE takes a causal test case
    and a causal specification and computes the causal effect of the intervention on the outcome of interest.
    """

    def setUp(self) -> None:
        # 1. Create Causal DAG
        temp_dir_path = create_temp_dir_if_non_existent()
        dag_dot_path = os.path.join(temp_dir_path, 'dag.dot')
        dag_dot = """digraph G { A -> C; D -> A; D -> C}"""
        f = open(dag_dot_path, 'w')
        f.write(dag_dot)
        f.close()
        self.causal_dag = CausalDAG(dag_dot_path)

        # 2. Create Scenario and Causal Specification
        A = Input("A", float)
        self.A = A
        C = Output("C", float)
        self.C = C
        D = Output("D", float)
        self.scenario = Scenario({A, C, D})
        self.causal_specification = CausalSpecification(scenario=self.scenario, causal_dag=self.causal_dag)

        # 3. Create a causal test case
        self.expected_causal_effect = ExactValue(4)
        self.causal_test_case = CausalTestCase(
            control_input_configuration={A: 0},
            expected_causal_effect=self.expected_causal_effect,
            treatment_input_configuration={A: 1},
            outcome_variables={C})

        # 4. Create dummy test data and write to csv
        np.random.seed(1)
        df = pd.DataFrame({'D': list(np.random.normal(60, 10, 1000))})  # D = exogenous
        df['A'] = [1 if d > 50 else 0 for d in df['D']]
        df['C'] = df['D'] + (4 * (df['A'] + 2))  # C = (4*(A+2)) + D
        self.observational_data_csv_path = os.path.join(temp_dir_path, 'observational_data.csv')
        df.to_csv(self.observational_data_csv_path, index=False)

        # 5. Create observational data collector
        # Obsolete?
        self.data_collector = ObservationalDataCollector(self.scenario, self.observational_data_csv_path)

        # 5. Create causal test engine
        self.causal_test_engine = CausalTestEngine(
            self.causal_test_case,
            self.causal_specification,
            self.data_collector
        )
        self.minimal_adjustment_set =\
            self.causal_test_engine.load_data()

        # 6. Easier to access treatment and outcome values
        self.treatment_value = 1
        self.control_value = 0

    def test_control_treatment_configs(self):
        test_case = CausalTestCase(
                    control_input_configuration={self.A: 0},
                    treatment_input_configuration={self.A: 1},
                    expected_causal_effect=self.expected_causal_effect,
                    outcome_variables={self.C}
                    )
        causal_test_engine = CausalTestEngine(
            test_case,
            self.causal_specification,
            self.data_collector
        )
        self.assertEqual(causal_test_engine.treatment_variables, [self.A])

    def test_positivity_violation_throws_exception(self):
        causal_test_engine = CausalTestEngine(
            self.causal_test_case,
            self.causal_specification,
            self.data_collector
        )
        causal_test_engine.load_data()
        causal_test_engine.scenario_execution_data_df.drop("A", axis=1, inplace=True)
        estimation_model = LinearRegressionEstimator(('A',),
                                                     self.treatment_value,
                                                     self.control_value,
                                                     self.minimal_adjustment_set,
                                                     ('C',),
                                                     self.causal_test_engine.scenario_execution_data_df)
        with self.assertRaises(Exception):
            causal_test_engine.execute_test(estimation_model)

    def test_check_no_positivity_violation(self):
        """ Check that no positivity violation is identified when there is no positivity violation. """
        variables_to_check = list(self.minimal_adjustment_set) + ['A'] + ['C']
        self.assertFalse(self.causal_test_engine._check_positivity_violation(variables_to_check))

    def test_check_positivity_violation_missing_confounder(self):
        """ Check that a positivity violation is identified when there is a positivity violation due to a missing
        confounder. """
        self.causal_test_engine.scenario_execution_data_df.drop(columns=['D'], inplace=True)  # Remove confounder
        variables_to_check = list(self.minimal_adjustment_set) + ['A'] + ['C']
        self.assertTrue(self.causal_test_engine._check_positivity_violation(variables_to_check))

    def test_check_positivity_violation_missing_treatment(self):
        """ Check that a positivity violation is identified when there is a positivity violation due to a missing
        treatment. """
        self.causal_test_engine.scenario_execution_data_df.drop(columns=['A'], inplace=True)  # Remove treatment
        variables_to_check = list(self.minimal_adjustment_set) + ['A'] + ['C']
        self.assertTrue(self.causal_test_engine._check_positivity_violation(variables_to_check))

    def test_check_positivity_violation_missing_outcome(self):
        """ Check that a positivity violation is identified when there is a positivity violation due to a missing
        outcome. """
        self.causal_test_engine.scenario_execution_data_df.drop(columns=['C'], inplace=True)  # Remove outcome
        variables_to_check = list(self.minimal_adjustment_set) + ['A'] + ['C']
        self.assertTrue(self.causal_test_engine._check_positivity_violation(variables_to_check))

    def test_execute_test_observational_causal_forest_estimator(self):
        """ Check that executing the causal test case returns the correct results for the dummy data using a causal
        forest estimator. """
        estimation_model = CausalForestEstimator(('A',),
                                                 self.treatment_value,
                                                 self.control_value,
                                                 self.minimal_adjustment_set,
                                                 ('C',),
                                                 self.causal_test_engine.scenario_execution_data_df)
        causal_test_result = self.causal_test_engine.execute_test(estimation_model)
        self.assertAlmostEqual(causal_test_result.ate, 4, delta=1)


    def test_invalid_causal_effect(self):
        """ Check that executing the causal test case returns the correct results for dummy data using a linear
        regression estimator. """
        causal_test_case = CausalTestCase(
            control_input_configuration={self.A: 0},
            expected_causal_effect=self.expected_causal_effect,
            treatment_input_configuration={self.A: 1},
            outcome_variables={self.C},
            effect="error")
        # 5. Create causal test engine
        causal_test_engine = CausalTestEngine(
            causal_test_case,
            self.causal_specification,
            self.data_collector
        )
        with self.assertRaises(Exception):
            causal_test_engine.load_data()


    def test_execute_test_observational_linear_regression_estimator(self):
        """ Check that executing the causal test case returns the correct results for dummy data using a linear
        regression estimator. """
        estimation_model = LinearRegressionEstimator(('A',),
                                                     self.treatment_value,
                                                     self.control_value,
                                                     self.minimal_adjustment_set,
                                                     ('C',),
                                                     self.causal_test_engine.scenario_execution_data_df)
        causal_test_result = self.causal_test_engine.execute_test(estimation_model)
        self.assertAlmostEqual(causal_test_result.ate, 4, delta=1e-10)


    def test_execute_test_observational_linear_regression_estimator_direct_effect(self):
        """ Check that executing the causal test case returns the correct results for dummy data using a linear
        regression estimator. """
        causal_test_case = CausalTestCase(
            control_input_configuration={self.A: 0},
            expected_causal_effect=self.expected_causal_effect,
            treatment_input_configuration={self.A: 1},
            outcome_variables={self.C},
            effect="direct")

        # 5. Create causal test engine
        causal_test_engine = CausalTestEngine(
            causal_test_case,
            self.causal_specification,
            self.data_collector
        )
        self.minimal_adjustment_set = causal_test_engine.load_data()

        # 6. Easier to access treatment and outcome values
        self.treatment_value = 1
        self.control_value = 0
        estimation_model = LinearRegressionEstimator(('A',),
                                                     self.treatment_value,
                                                     self.control_value,
                                                     self.minimal_adjustment_set,
                                                     ('C',),
                                                     causal_test_engine.scenario_execution_data_df)
        causal_test_result = causal_test_engine.execute_test(estimation_model)
        self.assertAlmostEqual(causal_test_result.ate, 0, delta=1e-10)

    def test_execute_test_observational_linear_regression_estimator_risk_ratio(self):
        """ Check that executing the causal test case returns the correct results for dummy data using a linear
        regression estimator. """
        estimation_model = LinearRegressionEstimator(('D',),
                                                     self.treatment_value,
                                                     self.control_value,
                                                     self.minimal_adjustment_set,
                                                     ('A',),
                                                     self.causal_test_engine.scenario_execution_data_df)
        causal_test_result = self.causal_test_engine.execute_test(estimation_model, estimate_type="risk_ratio")
        self.assertEqual(int(causal_test_result.ate), 0)


    def test_invalid_estimate_type(self):
        """ Check that executing the causal test case returns the correct results for dummy data using a linear
        regression estimator. """
        estimation_model = LinearRegressionEstimator(('D',),
                                                     self.treatment_value,
                                                     self.control_value,
                                                     self.minimal_adjustment_set,
                                                     ('A',),
                                                     self.causal_test_engine.scenario_execution_data_df)
        with self.assertRaises(ValueError):
            self.causal_test_engine.execute_test(estimation_model, estimate_type="invalid")

    def test_execute_test_observational_linear_regression_estimator_squared_term(self):
        """ Check that executing the causal test case returns the correct results for dummy data with a squared term
        using a linear regression estimator. C ~ 4*(A+2) + D + D^2"""
        estimation_model = LinearRegressionEstimator(('A',),
                                                     self.treatment_value,
                                                     self.control_value,
                                                     self.minimal_adjustment_set,
                                                     ('C',),
                                                     self.causal_test_engine.scenario_execution_data_df)
        estimation_model.add_squared_term_to_df('D')
        causal_test_result = self.causal_test_engine.execute_test(estimation_model)
        self.assertAlmostEqual(round(causal_test_result.ate, 1), 4, delta=1)

    def test_execute_observational_causal_forest_estimator_cates(self):
        """ Check that executing the causal test case returns the correct conditional average treatment effects for
        dummy data with effect multiplicative effect modification. C ~ (4*(A+2) + D)*M """
        # Add some effect modifier M that has a multiplicative effect on C
        self.causal_test_engine.scenario_execution_data_df['M'] = \
            np.random.randint(1, 5, len(self.causal_test_engine.scenario_execution_data_df))
        self.causal_test_engine.scenario_execution_data_df['C'] *= \
            self.causal_test_engine.scenario_execution_data_df['M']
        estimation_model = CausalForestEstimator(('A',),
                                                 self.treatment_value,
                                                 self.control_value,
                                                 self.minimal_adjustment_set,
                                                 ('C',),
                                                 self.causal_test_engine.scenario_execution_data_df,
                                                 effect_modifiers={Input('M', int): None})
        causal_test_result = self.causal_test_engine.execute_test(estimation_model, estimate_type='cate')
        causal_test_result = causal_test_result.ate
        # Check that each effect modifier's strata has a greater ATE than the last (ascending order)
        causal_test_result_m1 = causal_test_result.loc[causal_test_result['M'] == 1]
        causal_test_result_m2 = causal_test_result.loc[causal_test_result['M'] == 2]
        causal_test_result_m3 = causal_test_result.loc[causal_test_result['M'] == 3]
        causal_test_result_m4 = causal_test_result.loc[causal_test_result['M'] == 4]
        self.assertLess(causal_test_result_m1['cate'].mean(), causal_test_result_m2['cate'].mean())
        self.assertLess(causal_test_result_m2['cate'].mean(), causal_test_result_m3['cate'].mean())
        self.assertLess(causal_test_result_m3['cate'].mean(), causal_test_result_m4['cate'].mean())

    def tearDown(self) -> None:
        remove_temp_dir_if_existent()
