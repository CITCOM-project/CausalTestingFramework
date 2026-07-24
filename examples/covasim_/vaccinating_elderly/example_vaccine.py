import os
import logging
import pandas as pd
from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.testing.causal_test_case import CausalTestCase
from causal_testing.testing.causal_effect import Positive, Negative, NoEffect
from causal_testing.estimation.linear_regression_estimator import LinearRegressionEstimator


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format="%(message)s")
ROOT = os.path.realpath(os.path.dirname(__file__))


def run_test_case(verbose: bool = False):
    """Run the causal test case for the effect of changing vaccine to prioritise elderly from observational
    data that was previously simulated.

    :param verbose: Whether to print verbose details (causal test results).
    :return results_dict: A dictionary containing ATE, 95% CIs, and Test Pass/Fail
    """

    # Read in the Causal DAG
    causal_dag = CausalDAG(os.path.join(ROOT, "dag.dot"))

    # 5. Read the previously simulated data
    obs_df = pd.read_csv(os.path.join(ROOT, "simulated_data.csv"))

    # 6. Express expected outcomes
    expected_outcome_effects = {
        "cum_infections": Positive(),
        "cum_vaccinations": Negative(),
        "cum_vaccinated": Negative(),
        "max_doses": NoEffect(),
    }
    results_dict = {"cum_infections": {}, "cum_vaccinations": {}, "cum_vaccinated": {}, "max_doses": {}}

    for outcome_variable, expected_effect in expected_outcome_effects.items():
        causal_test_case = CausalTestCase(
            expected_causal_effect=expected_effect,
            effect_measure="ate",
            estimator=LinearRegressionEstimator(
                treatment_variable="vaccine",
                outcome_variable=outcome_variable,
                treatment_value=1,
                control_value=0,
                adjustment_set=causal_dag.identification(
                    treatment_variable="vaccine", outcome_variable=outcome_variable
                ),
            ),
        )

        causal_test_case.execute_test(obs_df)

        if verbose:
            logging.info("Causation:\n%s", causal_test_case.result)

        results_dict[outcome_variable]["ate"] = causal_test_case.result.effect_estimate.value

        results_dict[outcome_variable]["cis"] = [
            causal_test_case.result.effect_estimate.ci_low,
            causal_test_case.result.effect_estimate.ci_high,
        ]

        results_dict[outcome_variable]["test_passes"] = causal_test_case.expected_causal_effect.apply(
            causal_test_case.result.effect_estimate
        )

    return results_dict


def test_example_vaccine():
    run_test_case()


if __name__ == "__main__":

    test_results = run_test_case(verbose=True)

    logging.info("%s", test_results)
