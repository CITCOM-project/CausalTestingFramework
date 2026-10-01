import numpy as np

from causal_testing.visualisation.geometry import parse_dot_spline


def test_parse_dot_spline_e():
    np.testing.assert_array_equal(
        parse_dot_spline("e,168.34,180.1 168.34,215.7 168.34,207.98 168.34,198.71 168.34,190.11"),
        np.array(
            [
                (168.34, 215.7),
                (168.34, 207.98),
                (168.34, 198.71),
                (168.34, 190.11),
                (168.34, 180.1),
            ]
        ),
    )


def test_parse_dot_spline_s():
    np.testing.assert_array_equal(
        parse_dot_spline("s,168.34,215.7 168.34,207.98 168.34,198.71 168.34,190.11 168.34,180.1"),
        np.array(
            [
                (168.34, 215.7),
                (168.34, 207.98),
                (168.34, 198.71),
                (168.34, 190.11),
                (168.34, 180.1),
            ]
        ),
    )
