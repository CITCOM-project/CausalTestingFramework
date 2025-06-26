"""
Causal testing framework example script.

This example demonstrates the use of a custom EmpiricalMeanEstimator
for causal testing and runs causal tests using the framework.
"""

import os
from typing import Tuple, Optional, Any

from causal_testing.main import (
    CausalTestingFramework,
    CausalTestingPaths,
    setup_logging
)
from causal_testing.estimation.abstract_estimator import Estimator

setup_logging(verbose=True)


class EmpiricalMeanEstimator(Estimator):
    """
    Estimator that computes treatment effects using empirical means.

    This estimator calculates the Average Treatment Effect (ATE) and risk ratio
    by directly comparing the means of treatment and control groups.
    """

    def add_modelling_assumptions(self):
        """Add modeling assumptions for this estimator."""
        self.modelling_assumptions += (
            "The data must contain runs with the exact configuration of interest."
        )

    def estimate_ate(self) -> Tuple[float, Optional[Any]]:
        """
        Estimate the Average Treatment Effect.

        Returns:
            Tuple containing the ATE estimate and optional additional data.
        """
        control_results = self.df.where(
            self.df[self.base_test_case.treatment_variable.name] == self.control_value
        )[self.base_test_case.outcome_variable.name].dropna()

        treatment_results = self.df.where(
            self.df[self.base_test_case.treatment_variable.name] == self.treatment_value
        )[self.base_test_case.outcome_variable.name].dropna()

        return treatment_results.mean() - control_results.mean(), None

    def estimate_risk_ratio(self) -> Tuple[float, Optional[Any]]:
        """
        Estimate the risk ratio.

        Returns:
            Tuple containing the risk ratio estimate and optional additional data.
        """
        control_results = self.df.where(
            self.df[self.base_test_case.treatment_variable.name] == self.control_value
        )[self.base_test_case.outcome_variable.name].dropna()

        treatment_results = self.df.where(
            self.df[self.base_test_case.treatment_variable.name] == self.treatment_value
        )[self.base_test_case.outcome_variable.name].dropna()

        return treatment_results.mean() / control_results.mean(), None


ROOT = os.path.realpath(os.path.dirname(__file__))


def run_causal_tests() -> None:
    """
    Run causal tests using the framework.

    Sets up paths, initialises the framework, loads tests, runs them,
    and saves the results.
    """
    dag_path = os.path.join(ROOT, "dag.dot")
    data_paths = [os.path.join(ROOT, "data/random/data_random_1000.csv")]
    test_config_path = os.path.join(ROOT, "causal_tests.json")
    output_path = os.path.join(ROOT, "causal_test_results.json")

    paths = CausalTestingPaths(
        dag_path=dag_path,
        data_paths=data_paths,
        test_config_path=test_config_path,
        output_path=output_path
    )

    framework = CausalTestingFramework(paths=paths, ignore_cycles=False)
    framework.setup()
    framework.load_tests()
    results = framework.run_tests()
    framework.save_results(results)


if __name__ == "__main__":
    run_causal_tests()
