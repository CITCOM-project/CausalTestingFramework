import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from causal_testing.specification.variable import Input
from causal_testing.utils.validation import CausalValidator
from causal_testing.specification.capabilities import TreatmentSequence

from causal_testing.estimation.instrumental_variable_estimator import InstrumentalVariableEstimator


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
            treatment="X",
            treatment_value=None,
            control_value=None,
            adjustment_set=set(),
            outcome="Y",
            instrument="Z",
        )
        self.assertEqual(iv_estimator.estimate_coefficient(self.df), 2)

    def test_estimate_coefficient(self):
        """
        Test we get the correct coefficient.
        """
        iv_estimator = InstrumentalVariableEstimator(
            df=self.df,
            treatment="X",
            treatment_value=None,
            control_value=None,
            adjustment_set=set(),
            outcome="Y",
            instrument="Z",
        )
        coefficient, [low, high] = iv_estimator.estimate_coefficient()
        self.assertEqual(coefficient[0], 2)
