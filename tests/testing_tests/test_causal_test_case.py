import unittest
import os
from tests.test_helpers import create_temp_dir_if_non_existent, remove_temp_dir_if_existent
from causal_testing.specification.variable import Input, Output
from causal_testing.testing.causal_test_case import CausalTestCase
from causal_testing.testing.causal_test_outcome import ExactValue
from causal_testing.testing.base_test_case import BaseTestCase


class TestCausalTestCase(unittest.TestCase):
    """Test the CausalTestCase class.

    The base test case is a data class which contains the minimum information
    necessary to perform identification. The CausalTestCase class represents
    a causal test case. We here test the basic getter methods.
    """

    def setUp(self) -> None:
        # 2. Create Scenario and Causal Specification
        A = Input("A", float)
        C = Output("C", float)

        # 3. Create an intervention and causal test case
        self.expected_causal_effect = ExactValue(4)
        self.base_test_case = BaseTestCase(A, C)
        self.causal_test_case = CausalTestCase(
            base_test_case=self.base_test_case,
            expected_causal_effect=self.expected_causal_effect,
            control_value=0,
            treatment_value=1,
        )

    def test_get_treatment_variable(self):
        self.assertEqual(self.causal_test_case.get_treatment_variable(), "A")

    def test_get_outcome_variable(self):
        self.assertEqual(self.causal_test_case.get_outcome_variable(), "C")

    def test_get_treatment_value(self):
        self.assertEqual(self.causal_test_case.get_treatment_value(), 1)

    def test_get_control_value(self):
        self.assertEqual(self.causal_test_case.get_control_value(), 0)

    def test_str(self):
        self.assertEqual(
            str(self.causal_test_case),
            "Running {'A': 1} instead of {'A': 0} should cause the following changes to"
            " {Output: C::float}: ExactValue: 4±0.2.",
        )

    def tearDown(self) -> None:
        remove_temp_dir_if_existent()
