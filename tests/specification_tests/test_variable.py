import unittest
import os
import networkx as nx
from enum import Enum
import z3

from causal_testing.specification.variable import z3_types, Variable, Input, Output, Meta


class TestVariable(unittest.TestCase):

    """
    Test the CausalDAG class for creation of Causal Directed Acyclic Graphs (DAGs).

    In particular, confirm whether the Causal DAG class creates valid causal directed acyclic graphs (empty and directed
    graphs without cycles) and refuses to create invalid (cycle-containing) graphs.
    """

    def setUp(self) -> None:
        pass

    def test_z3_types_enum(self):
        class Color(Enum):
            RED = 1
            GREEN = 2
            BLUE = 3
        dtype, _ = z3.EnumSort("color", ("RED", "GREEN", "BLUE"))
        z3_color = z3.Const("color", dtype)
        expected = z3_types(Color)("color")
        # No actual way to assert their equality since they are two different objects
        expected_values = [expected.sort().constructor(c)() for c in range(expected.sort().num_constructors())]
        z3_color_values = [z3_color.sort().constructor(c)() for c in range(z3_color.sort().num_constructors())]

        # This isn't very good, but I think it's the best we can do since even
        # z3_types(Color)("color") != z3_types(Color)("color")
        self.assertEqual(list(map(str, expected_values)), list(map(str, z3_color_values)))

    def test_z3_types_invalid(self):
        with self.assertRaises(ValueError):
            class Err():
                pass
            z3_types(Err)


    def test_typestring(self):
        class Var(Variable):
            def copy(self):
                pass
        var = Var("v", int)
        self.assertEqual(var.typestring(), "Var")
