import os
import logging

import pandas as pd
from scipy.stats import bootstrap

from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.testing.causal_test_case import CausalTestCase
from causal_testing.testing.causal_effect import ExactValue, Positive
from causal_testing.estimation.linear_regression_estimator import LinearRegressionEstimator
from causal_testing.estimation.abstract_estimator import Estimator
from causal_testing.estimation.effect_estimate import EffectEstimate


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format="%(message)s")


class EmpiricalMeanEstimator(Estimator):
    """
    Custom estimator class to estimate the causal effect based on the empirical mean.
    """

    def add_modelling_assumptions(self):
        """
        Add modelling assumptions to the estimator. This is a list of strings which list the modelling assumptions that
        must hold if the resulting causal inference is to be considered valid.
        """
        self.modelling_assumptions += "The data must contain runs with the exact configuration of interest."

    def estimate_risk_ratio(self, df: pd.DataFrame) -> EffectEstimate:
        """Estimate the outcomes under control and treatment.
        :param df: The data to use.
        :return: The empirical average treatment effect.
        """

        control_results = df.where(df[self.treatment_variable] == self.control_value)[self.outcome_variable].dropna()
        treatment_results = df.where(df[self.treatment_variable] == self.treatment_value)[
            self.outcome_variable
        ].dropna()

        def risk_ratio(sample1, sample2):
            return sample1.mean() / sample2.mean()

        bootstraps = bootstrap((treatment_results, control_results), risk_ratio, confidence_level=self.alpha)
        return EffectEstimate(
            type="risk_ratio",
            value=risk_ratio(treatment_results, control_results),
            ci_low=bootstraps.confidence_interval.low,
            ci_high=bootstraps.confidence_interval.high,
        )


# Read in the Causal DAG
ROOT = os.path.realpath(os.path.dirname(__file__))
causal_dag = CausalDAG(f"{ROOT}/dag.dot")

OBSERVATIONAL_DATA_PATH = f"{ROOT}/data/random/data_random_1000.csv"


def test_poisson_intensity_num_shapes(save=False):
    intensity_num_shapes_results = []
    observational_df = pd.read_csv(OBSERVATIONAL_DATA_PATH, index_col=0).astype(float)
    causal_test_cases = [
        (
            CausalTestCase(
                treatment_variable="intensity",
                outcome_variable="num_shapes_unit",
                expected_causal_effect=ExactValue(4, atol=0.5),
                effect_measure="risk_ratio",
                estimator=EmpiricalMeanEstimator(
                    treatment_variable="intensity",
                    outcome_variable="num_shapes_unit",
                    treatment_value=treatment_value,
                    control_value=control_value,
                    adjustment_set=causal_dag.identification(
                        treatment_variable="intensity", outcome_variable="num_shapes_unit"
                    ),
                    alpha=0.05,
                ),
            ),
            f"{ROOT}/data/smt_100/data_smt_wh{wh}_100.csv",
            CausalTestCase(
                treatment_variable="intensity",
                outcome_variable="num_shapes_unit",
                expected_causal_effect=ExactValue(4, atol=0.5),
                effect_measure="risk_ratio",
                estimator=LinearRegressionEstimator(
                    treatment_variable="intensity",
                    outcome_variable="num_shapes_unit",
                    treatment_value=treatment_value,
                    control_value=control_value,
                    adjustment_set=causal_dag.identification(
                        treatment_variable="intensity", outcome_variable="num_shapes_unit"
                    ),
                    formula="num_shapes_unit ~ I(intensity ** 2) + intensity - 1",
                    alpha=0.05,
                ),
            ),
        )
        for control_value, treatment_value in [(1, 2), (2, 4), (4, 8), (8, 16)]
        for wh in range(1, 11)
    ]

    for smt_causal_test, datapath, obs_causal_test in causal_test_cases:
        smt_causal_test.execute_test(df=pd.read_csv(datapath, index_col=0).astype(float))
        obs_causal_test.execute_test(observational_df)

    intensity_num_shapes_results += [
        {
            "width": obs_causal_test.estimator.control_value,
            "height": obs_causal_test.estimator.treatment_value,
            "control": obs_causal_test.estimator.control_value,
            "treatment": obs_causal_test.estimator.treatment_value,
            "smt_risk_ratio": smt_causal_test.result.effect_estimate.value,
            "obs_risk_ratio": obs_causal_test.result.effect_estimate.value[0],
        }
        for smt_causal_test, _, obs_causal_test in causal_test_cases
    ]
    intensity_num_shapes_results = pd.DataFrame(intensity_num_shapes_results)
    if save:
        intensity_num_shapes_results.to_csv("intensity_num_shapes_results_random_1000.csv")
    logger.info("%s", intensity_num_shapes_results)


def test_poisson_width_num_shapes(save=False):
    df = pd.read_csv(OBSERVATIONAL_DATA_PATH, index_col=0).astype(float)
    causal_test_cases = [
        CausalTestCase(
            treatment_variable="width",
            outcome_variable="num_shapes_unit",
            expected_causal_effect=Positive(),
            effect_measure="ate_calculated",
            estimator=LinearRegressionEstimator(
                treatment_variable="width",
                outcome_variable="num_shapes_unit",
                treatment_value=w + 1.0,
                control_value=float(w),
                adjustment_set=causal_dag.identification(
                    treatment_variable="width",
                    outcome_variable="num_shapes_unit",
                ),
                adjustment_config={"intensity": i},
                formula="num_shapes_unit ~ width + I(intensity ** 2)+I(width ** -1)+intensity-1",
                alpha=0.05,
            ),
        )
        for i in range(1, 17)
        for w in range(1, 10)
    ]
    for test in causal_test_cases:
        test.execute_test(df)
    width_num_shapes_results = [
        {
            "control": causal_test.estimator.control_value,
            "treatment": causal_test.estimator.treatment_value,
            "intensity": causal_test.estimator.adjustment_config["intensity"],
            "ate": causal_test.result.effect_estimate.value[0],
            "ci_low": causal_test.result.effect_estimate.ci_low,
            "ci_high": causal_test.result.effect_estimate.ci_high,
        }
        for causal_test in causal_test_cases
    ]
    width_num_shapes_results = pd.DataFrame(width_num_shapes_results)
    if save:
        width_num_shapes_results.to_csv("width_num_shapes_results_random_1000.csv")
    logger.info("%s", width_num_shapes_results)


if __name__ == "__main__":
    test_poisson_intensity_num_shapes(save=False)
    test_poisson_width_num_shapes(save=True)
