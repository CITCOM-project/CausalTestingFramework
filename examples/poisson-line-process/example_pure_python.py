import os
import logging

import pandas as pd

from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.specification.scenario import Scenario
from causal_testing.specification.variable import Input, Output
from causal_testing.specification.causal_specification import CausalSpecification
from causal_testing.testing.causal_test_case import CausalTestCase
from causal_testing.testing.causal_test_outcome import ExactValue, Positive
from causal_testing.estimation.linear_regression_estimator import LinearRegressionEstimator
from causal_testing.estimation.abstract_estimator import Estimator
from causal_testing.testing.base_test_case import BaseTestCase
from causal_testing.testing.causal_test_suite import CausalTestSuite


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format="%(message)s")


class EmpiricalMeanEstimator(Estimator):
    def add_modelling_assumptions(self):
        """
        Add modelling assumptions to the estimator. This is a list of strings which list the modelling assumptions that
        must hold if the resulting causal inference is to be considered valid.
        """
        self.modelling_assumptions += "The data must contain runs with the exact configuration of interest."

    def estimate_ate(self) -> float:
        """Estimate the outcomes under control and treatment.
        :return: The empirical average treatment effect.
        """
        control_results = self.df.where(self.df[self.treatment] == self.control_value)[self.outcome].dropna()
        treatment_results = self.df.where(self.df[self.treatment] == self.treatment_value)[self.outcome].dropna()
        return treatment_results.mean() - control_results.mean(), None

    def estimate_risk_ratio(self) -> float:
        """Estimate the outcomes under control and treatment.
        :return: The empirical average treatment effect.
        """
        control_results = self.df.where(self.df[self.treatment] == self.control_value)[self.outcome].dropna()
        treatment_results = self.df.where(self.df[self.treatment] == self.treatment_value)[self.outcome].dropna()
        return treatment_results.mean() / control_results.mean(), None


# 1. Read in the Causal DAG
ROOT = os.path.realpath(os.path.dirname(__file__))
causal_dag = CausalDAG(f"{ROOT}/dag.dot")

# 2. Create variables
width = Input("width", float)
height = Input("height", float)
intensity = Input("intensity", float)

num_lines_abs = Output("num_lines_abs", float)
num_lines_unit = Output("num_lines_unit", float)
num_shapes_abs = Output("num_shapes_abs", float)
num_shapes_unit = Output("num_shapes_unit", float)

# 3. Create scenario
scenario = Scenario(
    variables={
        width,
        height,
        intensity,
        num_lines_abs,
        num_lines_unit,
        num_shapes_abs,
        num_shapes_unit,
    }
)

# 4. Construct a causal specification from the scenario and causal DAG
causal_specification = CausalSpecification(scenario, causal_dag)

observational_data_path = f"{ROOT}/data/random/data_random_1000.csv"


def causal_test_intensity_num_shapes(
    observational_data_path,
    causal_test_case,
    square_terms=[],
    inverse_terms=[],
    empirical=False,
):
    # 8. Set up an estimator
    data = pd.read_csv(observational_data_path, index_col=0).astype(float)

    treatment = causal_test_case.treatment_variable.name
    outcome = causal_test_case.outcome_variable.name

    estimator = None
    if empirical:
        estimator = EmpiricalMeanEstimator(
            treatment=[treatment],
            control_value=causal_test_case.control_value,
            treatment_value=causal_test_case.treatment_value,
            adjustment_set=set(),
            outcome=[outcome],
            df=data,
            effect_modifiers=causal_test_case.effect_modifier_configuration,
        )
    else:
        square_terms = [f"I({t} ** 2)" for t in square_terms]
        inverse_terms = [f"I({t} ** -1)" for t in inverse_terms]
        estimator = LinearRegressionEstimator(
            treatment=treatment,
            control_value=causal_test_case.control_value,
            treatment_value=causal_test_case.treatment_value,
            adjustment_set=set(),
            outcome=outcome,
            df=data,
            effect_modifiers=causal_test_case.effect_modifier_configuration,
            formula=f"{outcome} ~ {treatment} + {'+'.join(square_terms + inverse_terms + list([e for e in causal_test_case.effect_modifier_configuration]))} -1",
        )

    # 9. Execute the test
    causal_test_result = causal_test_case.execute_test(estimator)

    return causal_test_result


def test_poisson_intensity_num_shapes(save=False):
    intensity_num_shapes_results = []
    base_test_case = BaseTestCase(treatment_variable=intensity, outcome_variable=num_shapes_unit)
    for wh in range(1, 11):
        smt_data_path = f"{ROOT}/data/smt_100/data_smt_wh{wh}_100.csv"
        causal_test_case_list = [
            CausalTestCase(
                base_test_case=base_test_case,
                expected_causal_effect=ExactValue(4, atol=0.5),
                treatment_value=treatment_value,
                control_value=control_value,
                estimate_type="risk_ratio",
            )
            for control_value, treatment_value in [(1, 2), (2, 4), (4, 8), (8, 16)]
        ]
        test_suite = CausalTestSuite()
        test_suite.add_test_object(
            base_test_case,
            causal_test_case_list=causal_test_case_list,
            estimators=[LinearRegressionEstimator, EmpiricalMeanEstimator],
        )
        test_suite_results = test_suite.execute_test_suite(
            causal_specification, pd.read_csv(smt_data_path, index_col=0).astype(float)
        )

        smt_risk_ratios = [
            causal_test_result.test_value.value
            for causal_test_result in test_suite_results[base_test_case]["EmpiricalMeanEstimator"]
        ]

        intensity_num_shapes_results += [
            {
                "width": wh,
                "height": wh,
                "control": obs_causal_test_result.estimator.control_value,
                "treatment": obs_causal_test_result.estimator.treatment_value,
                "smt_risk_ratio": smt_causal_test_result.test_value.value,
                "obs_risk_ratio": obs_causal_test_result.test_value.value[0],
            }
            for obs_causal_test_result, smt_causal_test_result in zip(
                test_suite_results[base_test_case]["LinearRegressionEstimator"],
                test_suite_results[base_test_case]["EmpiricalMeanEstimator"],
            )
        ]
    intensity_num_shapes_results = pd.DataFrame(intensity_num_shapes_results)
    if save:
        intensity_num_shapes_results.to_csv("intensity_num_shapes_results_random_1000.csv")
    logger.info("%s", intensity_num_shapes_results)


def test_poisson_width_num_shapes(save=False):
    base_test_case = BaseTestCase(treatment_variable=width, outcome_variable=num_shapes_unit)
    causal_test_case_list = [
        CausalTestCase(
            base_test_case=base_test_case,
            expected_causal_effect=Positive(),
            control_value=float(w),
            treatment_value=w + 1.0,
            estimate_type="ate_calculated",
            effect_modifier_configuration={"intensity": i},
        )
        for i in range(17)
        for w in range(1, 10)
    ]
    test_suite = CausalTestSuite()
    test_suite.add_test_object(
        base_test_case,
        causal_test_case_list=causal_test_case_list,
        estimators=[LinearRegressionEstimator],
    )
    test_suite_results = test_suite.execute_test_suite(
        causal_specification, pd.read_csv(observational_data_path, index_col=0).astype(float)
    )
    width_num_shapes_results = [
        {
            "control": causal_test_result.estimator.control_value,
            "treatment": causal_test_result.estimator.treatment_value,
            "intensity": causal_test_result.effect_modifier_configuration["intensity"],
            "ate": causal_test_result.test_value.value[0],
            "ci_low": causal_test_result.confidence_intervals[0][0],
            "ci_high": causal_test_result.confidence_intervals[1][0],
        }
        for causal_test_result in test_suite_results[base_test_case]["LinearRegressionEstimator"]
    ]
    width_num_shapes_results = pd.DataFrame(width_num_shapes_results)
    if save:
        width_num_shapes_results.to_csv("width_num_shapes_results_random_1000.csv")
    logger.info("%s", width_num_shapes_results)


if __name__ == "__main__":
    test_poisson_intensity_num_shapes(save=False)
    # test_poisson_width_num_shapes(save=True)
