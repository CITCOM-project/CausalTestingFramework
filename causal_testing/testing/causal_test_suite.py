"""This module contains the CausalTestSuite class, for details on using it:
https://causal-testing-framework.readthedocs.io/en/latest/test_suite.html"""
from collections import UserDict
from typing import Type, Iterable
from causal_testing.testing.base_test_case import BaseTestCase
from causal_testing.testing.causal_test_case import CausalTestCase
from causal_testing.testing.estimators import Estimator


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
