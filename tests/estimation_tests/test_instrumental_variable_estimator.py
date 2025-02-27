import unittest
import pandas as pd
import numpy as np

from causal_testing.estimation.instrumental_variable_estimator import InstrumentalVariableEstimator
from causal_testing.testing.base_test_case import BaseTestCase
from causal_testing.specification.variable import Input, Output


class TestInstrumentalVariableEstimator(unittest.TestCase):
    """
    Test the instrumental variable estimator.
    """

    @classmethod
    def setUpClass(cls) -> None:
        Z = np.linspace(0, 10)
        X = 2 * Z
        Y = 2 * X
        cls.df = pd.DataFrame({"Z": Z, "X": X, "Y": Y})

    def test_estimate_coefficient(self):
        """
        Test we get the correct coefficient.
        """
        iv_estimator = InstrumentalVariableEstimator(
            df=self.df,
            base_test_case=BaseTestCase(Input("X", float), Output("Y", float)),
            treatment_value=None,
            control_value=None,
            adjustment_set=set(),
            instrument="Z",
        )
        coefficient, [low, high] = iv_estimator.estimate_coefficient()
        self.assertEqual(coefficient[0], 2)
        self.assertEqual(low[0], 2)
        self.assertEqual(high[0], 2)
