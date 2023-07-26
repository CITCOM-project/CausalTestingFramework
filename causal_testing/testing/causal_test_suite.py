"""This module contains the CausalTestSuite class, for details on using it:
https://causal-testing-framework.readthedocs.io/en/latest/test_suite.html"""
import logging

from collections import UserDict
from typing import Type, Iterable
from causal_testing.testing.base_test_case import BaseTestCase
from causal_testing.testing.causal_test_case import CausalTestCase
from causal_testing.testing.estimators import Estimator
from causal_testing.testing.causal_test_result import CausalTestResult, TestValue
from causal_testing.data_collection.data_collector import ObservationalDataCollector
from causal_testing.specification.causal_specification import CausalSpecification

logger = logging.getLogger(__name__)


class CausalTestSuite(UserDict):
    """
    A CausalTestSuite is an extension of the UserDict class, therefore it behaves as a python dictionary with the added
    functionality of this class.
    The dictionary structure should be the keys are base_test_cases representing the treatment and outcome Variables,
    and the values are test objects. Test Objects hold a causal_test_case_list which is a list of causal_test_cases
    which provide control and treatment values, and an iterator of Estimator Class References

    This dictionary can be fed into the CausalTestEngines execute_test_suite function which will iterate over all the
    base_test_case's and execute each causal_test_case with each iterator.
    """

    def add_test_object(
            self,
            base_test_case: BaseTestCase,
            causal_test_case_list: Iterable[CausalTestCase],
            estimators_classes: Iterable[Type[Estimator]],
            estimate_type: str = "ate",
    ):
        """
        A setter object to allow for the easy construction of the dictionary test suite structure

        :param base_test_case: A BaseTestCase object consisting of a treatment variable, outcome variable and effect
        :param causal_test_case_list: A list of causal test cases to be executed
        :param estimators_classes: A list of estimator class references, the execute_test_suite function in the
            TestEngine will produce a list of test results for each estimator
        :param estimate_type: A string which denotes the type of estimate to return
        """
        test_object = {"tests": causal_test_case_list, "estimators": estimators_classes, "estimate_type": estimate_type}
        self.data[base_test_case] = test_object

    def execute_test_suite(self, data_collector: ObservationalDataCollector,
                           causal_specification: CausalSpecification) -> list[CausalTestResult]:
        """Execute a suite of causal tests and return the results in a list
        :param test_suite: CasualTestSuite object
        :return: A dictionary where each key is the name of the estimators specified and the values are lists of
                causal_test_result objects
        """
        if data_collector.data.empty:
            raise ValueError("No data has been loaded. Please call load_data prior to executing a causal test case.")
        data_collector.collect_data()
        test_suite_results = {}
        for edge in self:
            logger.info("treatment: %s", edge.treatment_variable)
            logger.info("outcome: %s", edge.outcome_variable)
            minimal_adjustment_set = causal_specification.causal_dag.identification(edge)
            minimal_adjustment_set = minimal_adjustment_set - set(edge.treatment_variable.name)
            minimal_adjustment_set = minimal_adjustment_set - set(edge.outcome_variable.name)

            variables_for_positivity = list(minimal_adjustment_set) + [
                edge.treatment_variable.name,
                edge.outcome_variable.name,
            ]

            if self._check_positivity_violation(variables_for_positivity, causal_specification.scenario, data_collector.data):
                raise ValueError("POSITIVITY VIOLATION -- Cannot proceed.")

            estimators = self[edge]["estimators"]
            tests = self[edge]["tests"]
            results = {}
            for estimator_class in estimators:
                causal_test_results = []

                for test in tests:
                    estimator = estimator_class(
                        test.treatment_variable.name,
                        test.treatment_value,
                        test.control_value,
                        minimal_adjustment_set,
                        test.outcome_variable.name,
                    )
                    if estimator.df is None:
                        estimator.df = data_collector.data
                    causal_test_result = self._return_causal_test_results(estimator, test)
                    causal_test_results.append(causal_test_result)

                results[estimator_class.__name__] = causal_test_results
            test_suite_results[edge] = results
        return test_suite_results

    def _return_causal_test_results(self, estimator, causal_test_case):
        """Depending on the estimator used, calculate the 95% confidence intervals and return in a causal_test_result

        :param estimator: An Estimator class object
        :param causal_test_case: The concrete test case to be executed
        :return: a CausalTestResult object containing the confidence intervals
        """
        if not hasattr(estimator, f"estimate_{causal_test_case.estimate_type}"):
            raise AttributeError(f"{estimator.__class__} has no {causal_test_case.estimate_type} method.")
        estimate_effect = getattr(estimator, f"estimate_{causal_test_case.estimate_type}")
        effect, confidence_intervals = estimate_effect(**causal_test_case.estimate_params)
        causal_test_result = CausalTestResult(
            estimator=estimator,
            test_value=TestValue(causal_test_case.estimate_type, effect),
            effect_modifier_configuration=causal_test_case.effect_modifier_configuration,
            confidence_intervals=confidence_intervals,
        )

        return causal_test_result

    def _check_positivity_violation(self, variables_list, scenario, data):
        """Check whether the dataframe has a positivity violation relative to the specified variables list.

        A positivity violation occurs when there is a stratum of the dataframe which does not have any data. Put simply,
        if we split the dataframe into covariate sub-groups, each sub-group must contain both a treated and untreated
        individual. If a positivity violation occurs, causal inference is still possible using a properly specified
        parametric estimator. Therefore, we should not throw an exception upon violation but raise a warning instead.

        :param variables_list: The list of variables for which positivity must be satisfied.
        :return: True if positivity is violated, False otherwise.
        """
        if not (set(variables_list) - {x.name for x in scenario.hidden_variables()}).issubset(
                data.columns
        ):
            missing_variables = set(variables_list) - set(data.columns)
            logger.warning(
                "Positivity violation: missing data for variables %s.\n"
                "Causal inference is only valid if a well-specified parametric model is used.\n"
                "Alternatively, consider restricting analysis to executions without the variables:"
                ".",
                missing_variables,
            )
            return True

        return False
