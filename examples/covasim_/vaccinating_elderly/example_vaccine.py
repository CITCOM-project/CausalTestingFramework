import os
import logging
import pandas as pd
from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.specification.variable import Input, Output
from causal_testing.testing.causal_test_case import CausalTestCase
from causal_testing.testing.causal_effect import Positive, Negative, NoEffect
from causal_testing.estimation.linear_regression_estimator import LinearRegressionEstimator
from causal_testing.testing.base_test_case import BaseTestCase


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format="%(message)s")
ROOT = os.path.realpath(os.path.dirname(__file__))


def run_test_case(verbose: bool = False):
    """Run the causal test case for the effect of changing vaccine to prioritise elderly from observational
    data that was previously simulated.

    :param verbose: Whether to print verbose details (causal test results).
    :return results_dict: A dictionary containing ATE, 95% CIs, and Test Pass/Fail
    """

    # 1. Read in the Causal DAG
    causal_dag = CausalDAG(os.path.join(ROOT, "dag.dot"))

    # 2. Create variables
    vaccine = Input("vaccine", int)
    cum_infections = Output("cum_infections", int)
    cum_vaccinations = Output("cum_vaccinations", int)
    cum_vaccinated = Output("cum_vaccinated", int)
    max_doses = Output("max_doses", int)

    # 5. Read the previously simulated data
    obs_df = pd.read_csv(os.path.join(ROOT, "simulated_data.csv"))

    # 6. Express expected outcomes
    expected_outcome_effects = {
        cum_infections: Positive(),
        cum_vaccinations: Negative(),
        cum_vaccinated: Negative(),
        max_doses: NoEffect(),
    }
    results_dict = {"cum_infections": {}, "cum_vaccinations": {}, "cum_vaccinated": {}, "max_doses": {}}

    for outcome_variable, expected_effect in expected_outcome_effects.items():
        base_test_case = BaseTestCase(treatment_variable=vaccine, outcome_variable=outcome_variable)
        causal_test_case = CausalTestCase(
            base_test_case=base_test_case,
            expected_causal_effect=expected_effect,
        )
        # 7. Obtain the minimal adjustment set for the causal test case from the causal DAG
        minimal_adjustment_set = causal_dag.identification(base_test_case)

        # 8. Build statistical model using the Linear Regression estimator
        linear_regression_estimator = LinearRegressionEstimator(
            base_test_case=base_test_case,
            treatment_value=1,
            control_value=0,
            adjustment_set=minimal_adjustment_set,
            df=obs_df,
        )

        # 9. Execute test and save results in dict
        causal_test_result = causal_test_case.execute_test(linear_regression_estimator)

        if verbose:
            logging.info("Causation:\n%s", causal_test_result)

        results_dict[outcome_variable.name]["ate"] = causal_test_result.effect_estimate.value

        results_dict[outcome_variable.name]["cis"] = [
            causal_test_result.effect_estimate.ci_low,
            causal_test_result.effect_estimate.ci_high,
        ]

        results_dict[outcome_variable.name]["test_passes"] = causal_test_case.expected_causal_effect.apply(
            causal_test_result
        )

    return results_dict


def test_example_vaccine():
    run_test_case()


if __name__ == "__main__":

    test_results = run_test_case(verbose=True)

    logging.info("%s", test_results)
