import unittest

import deap
import pandas as pd

from causal_testing.estimation.genetic_programming_regression_fitter import GP, mut_insert


def root(x):
    return x**0.5


class TestGP(unittest.TestCase):
    def test_init_invalid_fun_name(self):
        """
        Test that GP raises ValueError if sympy conversions are provided for invalid function names.
        """
        with self.assertRaises(ValueError):
            GP(df=pd.DataFrame(), features=[], outcome="", max_order=2, sympy_conversions={"power_1": ""})

    def test_simplify_string(self):
        """
        Test GP simplification
        """
        gp = GP(
            df=None,
            features=["x1"],
            outcome=None,
            max_order=1,
        )
        self.assertEqual(str(gp.simplify("power_1(x1)")), "x1")

    def test_fitness(self):
        """
        Test GP fitness function for perfect expression.
        """
        gp = GP(
            df=pd.DataFrame({"x1": [1, 2, 3], "outcome": [2, 3, 4]}),
            features=["x1"],
            outcome="outcome",
            max_order=0,
        )
        self.assertEqual(gp.fitness("add(x1, 1)"), (0,))

    def test_fitness_inf(self):
        """
        Test that GP returns infinity fitness for incalculable expressions.
        """
        gp = GP(
            df=pd.DataFrame({"x1": [1, 2, 3], "outcome": [2, 3, 4]}),
            features=["x1"],
            outcome="outcome",
            max_order=0,
            extra_operators=[(root, 1)],
        )
        self.assertEqual(gp.fitness("root(-1)"), (float("inf"),))

    def test_mut_insert_no_primitives(self):
        """Test that mut_insert returns the unmodified expression if there are no
        primitives of the appropriate type."""
        pset = deap.gp.PrimitiveSet("MAIN", 1)
        pset.addPrimitive(lambda x1, x2: x1 + x2, 1, name="add")
        expression = deap.gp.PrimitiveTree.from_string("add(ARG0, 1)", pset)
        self.assertEqual(
            mut_insert(
                expression,
                deap.gp.PrimitiveSet("MAIN", 1),
            ),
            (expression,),
        )
