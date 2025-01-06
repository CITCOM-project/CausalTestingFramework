import unittest
import pandas as pd
from operator import sub

from causal_testing.estimation.genetic_programming_regression_fitter import GP


class TestGP(unittest.TestCase):
    def test_init_invalid_fun_name(self):
        with self.assertRaises(ValueError):
            GP(df=pd.DataFrame(), features=[], outcome="", max_order=2, sympy_conversions={"power_1": ""})

    def test_simplify_string(self):
        gp = GP(
            df=None,
            features=["x1"],
            outcome=None,
            max_order=1,
        )
        self.assertEquals(str(gp.simplify("power_1(x1)")), "x1")

    def test_fitness_string(self):
        gp = GP(
            df=pd.DataFrame({"x1": [1, 2, 3], "outcome": [2, 3, 4]}),
            features=["x1"],
            outcome="outcome",
            max_order=0,
        )
        self.assertEquals(gp.fitness("add(x1, 1)"), (0,))
