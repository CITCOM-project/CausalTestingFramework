import unittest
from enum import Enum
import z3
from scipy.stats import norm, kstest

from causal_testing.specification.variable import z3_types, Variable, Input


class TestVariable(unittest.TestCase):

    """
    Test the Variable class for basic methods.
    """

    def setUp(self) -> None:
        pass

    def test_z3_types_enum(self):
        class Color(Enum):
            """
            Example enum class color.
            """

            RED = "RED"
            GREEN = "GREEN"
            BLUE = "BLUE"

        dtype, _ = z3.EnumSort("color", ("RED", "GREEN", "BLUE"))
        z3_color = z3.Const("color", dtype)
        expected = z3_types(Color)("color")
        # No actual way to assert their equality since they are two different objects
        expected_values = [expected.sort().constructor(c)() for c in range(expected.sort().num_constructors())]
        z3_color_values = [z3_color.sort().constructor(c)() for c in range(z3_color.sort().num_constructors())]

        # This isn't very good, but I think it's the best we can do since even
        # z3_types(Color)("color") != z3_types(Color)("color")
        self.assertEqual(list(map(str, expected_values)), list(map(str, z3_color_values)))

    def test_cast_z3_bool(self):
        bip = Input("bip", bool)
        s = z3.Solver()
        t = z3.Bool("t")
        f = z3.Bool("f")
        s.add(t)
        s.add(z3.Not(f))
        s.check()
        self.assertEqual(bip.cast(s.model()[t]), True)
        self.assertEqual(bip.cast(s.model()[f]), False)

    def test_cast_z3_string(self):
        ip = Input("bip", str)
        s = z3.Solver()
        t = z3.String("t")
        s.add(t == "hello")
        s.check()
        self.assertEqual(ip.cast(s.model()[t]), "hello")

    def test_sample_flakey(self):
        ip = Input("ip", float, norm)
        self.assertGreater(kstest(ip.sample(10), norm.cdf).pvalue, 0.95)

    def test_cast_enum(self):
        class Color(Enum):
            """
            Example enum class color.
            """

            RED = "RED"
            GREEN = "GREEN"
            BLUE = "BLUE"

        color = Input("color", Color)

        dtype, colours = z3.EnumSort("color", ("RED", "GREEN", "BLUE"))
        self.assertEqual(color.cast(colours[0]), Color.RED)

    def test_z3_value_enum(self):
        class Color(Enum):
            """
            Example enum class color.
            """

            RED = "RED"
            GREEN = "GREEN"
            BLUE = "BLUE"

        dtype, members = z3.EnumSort("color", ("RED", "GREEN", "BLUE"))
        z3_color = z3.Const("color", dtype)
        color = Input("color", Color)

        self.assertEqual(color.z3_val(z3_color, "RED"), members[0])

    def test_z3_types_custom(self):
        class Color:
            """
            Example enum class color.
            """

            RED = 1
            GREEN = 2
            BLUE = 3

            @classmethod
            def to_z3(cls):
                dtype, _ = z3.EnumSort("Color", ("RED", "GREEN", "BLUE"))
                return lambda x: z3.Const(x, dtype)

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

            class Err:
                """
                The simplest class which will elicit the correct error.
                """

            z3_types(Err)

    def test_typestring(self):
        class Var(Variable):
            pass

        var = Var("v", int)
        self.assertEqual(var.typestring(), "Var")

    def test_copy(self):
        ip = Input("ip", float, norm)
        self.assertTrue(ip.copy() is not ip)
        self.assertEqual(ip.copy().name, ip.name)
        self.assertEqual(ip.copy().datatype, ip.datatype)
        self.assertEqual(ip.copy().distribution, ip.distribution)
        self.assertEqual(repr(ip), repr(ip.copy()))


class TestZ3Methods(unittest.TestCase):

    """
    Test the Variable class for Z3 methods.

    TODO: These are all pretty hacky, to be honest, but Z3 makes checking this sort of thing really difficult.
    """

    def setUp(self) -> None:
        self.i1 = Input("i1", int)

    def test_ge_self(self):
        self.assertEqual(str(self.i1 >= self.i1), "i1 >= i1")

    def test_add(self):
        self.assertEqual(str(self.i1 + 1), "i1 + 1")

    def test_ge(self):
        self.assertEqual(str(self.i1 >= 5), "i1 >= 5")

    def test_mod(self):
        self.assertEqual(str(self.i1 % 2), "i1%2")

    def test_ne(self):
        self.assertEqual(str(self.i1 != 5), "i1 != 5")

    def test_neg(self):
        self.assertEqual(str(-self.i1), "-i1")

    def test_pow(self):
        self.assertEqual(str(self.i1**5), "i1**5")

    def test_le(self):
        self.assertEqual(str(self.i1 <= 5), "i1 <= 5")

    def test_mul(self):
        self.assertEqual(str(self.i1 * 2), "i1*2")

    def test_gt(self):
        self.assertEqual(str(self.i1 > 5), "i1 > 5")

    def test_truediv(self):
        self.assertEqual(str(self.i1 / 3), "i1/3")

    def test_sub(self):
        self.assertEqual(str(self.i1 - 4), "i1 - 4")

    def test_lt(self):
        self.assertEqual(str(self.i1 < 5), "i1 < 5")
