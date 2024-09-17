import unittest
import pandas as pd

from causal_testing.estimation.genetic_programming_regression_fitter import GP


class TestGP(unittest.TestCase):
    def test_init_invalid_fun_name(self):
        with self.assertRaises(ValueError):
            GP(df=pd.DataFrame(), features=[], outcome="", max_order=2, sympy_conversions={"power_1": ""})
