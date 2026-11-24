"""
This module tests the genetic programming regression fitter.
"""

import pandas as pd
import pytest
from deap.gp import PrimitiveSet, PrimitiveTree

from causal_testing.estimation.genetic_programming_regression_fitter import GP, mut_insert


def root(x):
    return x**0.5


def test_init_invalid_fun_name():
    """
    Test that GP raises ValueError if sympy conversions are provided for invalid function names.
    """
    with pytest.raises(ValueError):
        GP(df=pd.DataFrame(), features=[], outcome="", max_order=2, sympy_conversions={"power_1": ""})


@pytest.fixture(name="gp")
def _gp():
    """
    Fresh GP instance for each test, fitted to outcome = x1 + 1.
    """
    return GP(
        df=pd.DataFrame({"x1": [1, 2, 3], "outcome": [2, 3, 4]}),
        features=["x1"],
        outcome="outcome",
        max_order=1,
        extra_operators=[(root, 1)],
    )


def test_simplify_string(gp):
    """
    Test GP simplification
    """
    assert str(gp.simplify("power_1(x1)")) == "x1"


def test_fitness(gp):
    """
    Test GP fitness function for perfect expression.
    """
    assert gp.fitness("add(x1, 1)") == (0,)


def test_fitness_inf(gp):
    """
    Test that GP returns infinity fitness for incalculable expressions.
    """
    assert gp.fitness("root(-1)") == (float("inf"),)


def test_mut_insert_no_primitives():
    """Test that mut_insert returns the unmodified expression if there are no
    primitives of the appropriate type."""
    pset = PrimitiveSet("MAIN", 1)
    pset.addPrimitive(lambda x1, x2: x1 + x2, 1, name="add")
    expression = PrimitiveTree.from_string("add(ARG0, 1)", pset)
    assert mut_insert(
        expression,
        PrimitiveSet("MAIN", 1),
    ) == (expression,)
