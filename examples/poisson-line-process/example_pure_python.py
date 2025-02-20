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
        control_results = self.df.where(self.df[self.base_test_case.treatment_variable.name] == self.control_value)[
            self.base_test_case.outcome_variable.name
        ].dropna()
        treatment_results = self.df.where(self.df[self.base_test_case.treatment_variable.name] == self.treatment_value)[
            self.base_test_case.outcome_variable.name
        ].dropna()
        return treatment_results.mean() - control_results.mean(), None

    def estimate_risk_ratio(self) -> float:
        """Estimate the outcomes under control and treatment.
        :return: The empirical average treatment effect.
        """
        control_results = self.df.where(self.df[self.base_test_case.treatment_variable.name] == self.control_value)[
            self.base_test_case.outcome_variable.name
        ].dropna()
        treatment_results = self.df.where(self.df[self.base_test_case.treatment_variable.name] == self.treatment_value)[
            self.base_test_case.outcome_variable.name
        ].dropna()
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


def test_poisson_intensity_num_shapes(save=False):
    intensity_num_shapes_results = []
    base_test_case = BaseTestCase(treatment_variable=intensity, outcome_variable=num_shapes_unit)
    observational_df = pd.read_csv(observational_data_path, index_col=0).astype(float)
    causal_test_cases = [
        (
            CausalTestCase(
                base_test_case=base_test_case,
                expected_causal_effect=ExactValue(4, atol=0.5),
                estimate_type="risk_ratio",
                estimator=EmpiricalMeanEstimator(
                    base_test_case=base_test_case,
                    treatment_value=treatment_value,
                    control_value=control_value,
                    adjustment_set=causal_specification.causal_dag.identification(base_test_case),
                    df=pd.read_csv(f"{ROOT}/data/smt_100/data_smt_wh{wh}_100.csv", index_col=0).astype(float),
                    effect_modifiers=None,
                    alpha=0.05,
                    query="",
                ),
            ),
            CausalTestCase(
                base_test_case=base_test_case,
                expected_causal_effect=ExactValue(4, atol=0.5),
                estimate_type="risk_ratio",
                estimator=LinearRegressionEstimator(
                    base_test_case=base_test_case,
                    treatment_value=treatment_value,
                    control_value=control_value,
                    adjustment_set=causal_specification.causal_dag.identification(base_test_case),
                    df=observational_df,
                    effect_modifiers=None,
                    formula="num_shapes_unit ~ I(intensity ** 2) + intensity - 1",
                    alpha=0.05,
                    query="",
                ),
            ),
        )
        for control_value, treatment_value in [(1, 2), (2, 4), (4, 8), (8, 16)]
        for wh in range(1, 11)
    ]

    test_results = [(smt.execute_test(), observational.execute_test()) for smt, observational in causal_test_cases]

    intensity_num_shapes_results += [
        {
            "width": obs_causal_test_result.estimator.control_value,
            "height": obs_causal_test_result.estimator.treatment_value,
            "control": obs_causal_test_result.estimator.control_value,
            "treatment": obs_causal_test_result.estimator.treatment_value,
            "smt_risk_ratio": smt_causal_test_result.test_value.value,
            "obs_risk_ratio": obs_causal_test_result.test_value.value[0],
        }
        for smt_causal_test_result, obs_causal_test_result in test_results
    ]
    intensity_num_shapes_results = pd.DataFrame(intensity_num_shapes_results)
    if save:
        intensity_num_shapes_results.to_csv("intensity_num_shapes_results_random_1000.csv")
    logger.info("%s", intensity_num_shapes_results)


def test_poisson_width_num_shapes(save=False):
    base_test_case = BaseTestCase(treatment_variable=width, outcome_variable=num_shapes_unit)
    df = pd.read_csv(observational_data_path, index_col=0).astype(float)
    causal_test_cases = [
        CausalTestCase(
            base_test_case=base_test_case,
            expected_causal_effect=Positive(),
            estimate_type="ate_calculated",
            effect_modifier_configuration={"intensity": i},
            estimator=LinearRegressionEstimator(
                base_test_case=base_test_case,
                treatment_value=w + 1.0,
                control_value=float(w),
                adjustment_set=causal_specification.causal_dag.identification(base_test_case),
                df=df,
                effect_modifiers={"intensity": i},
                formula="num_shapes_unit ~ width + I(intensity ** 2)+I(width ** -1)+intensity-1",
                alpha=0.05,
            ),
        )
        for i in range(1, 17)
        for w in range(1, 10)
    ]
    test_results = [test.execute_test() for test in causal_test_cases]
    width_num_shapes_results = [
        {
            "control": causal_test_result.estimator.control_value,
            "treatment": causal_test_result.estimator.treatment_value,
            "intensity": causal_test_result.effect_modifier_configuration["intensity"],
            "ate": causal_test_result.test_value.value[0],
            "ci_low": causal_test_result.confidence_intervals[0][0],
            "ci_high": causal_test_result.confidence_intervals[1][0],
        }
        for causal_test_result in test_results
    ]
    width_num_shapes_results = pd.DataFrame(width_num_shapes_results)
    if save:
        width_num_shapes_results.to_csv("width_num_shapes_results_random_1000.csv")
    logger.info("%s", width_num_shapes_results)


if __name__ == "__main__":
    test_poisson_intensity_num_shapes(save=False)
    test_poisson_width_num_shapes(save=True)
