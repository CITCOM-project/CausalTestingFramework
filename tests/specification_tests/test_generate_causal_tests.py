"""
This module tests the generation of causal tests from a CausalDAG.
"""

import re

import pytest

from causal_testing.estimation.linear_regression_estimator import LinearRegressionEstimator
from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.testing.causal_effect import NoEffect, SomeEffect
from causal_testing.testing.causal_test_case import CausalTestCase


def sort_test_dict(test: dict):
    """
    Function for sorting lists of tests to check equivalence.
    """
    return test["name"]


@pytest.fixture(name="dag_path")
def _dag_path(tmp_path):
    """
    Path to an acyclic DAG with inputs X1, X2, X3, mediators Z and M, and outcome Y.
    """
    path = tmp_path / "dag.dot"
    path.write_text("digraph DAG { rankdir=LR; X1 -> Z; Z -> M; M -> Y; X2 -> Z; X3 -> M;}")
    return str(path)


@pytest.fixture(name="dcg_path")
def _dcg_path(tmp_path):
    """
    Path to a directed graph containing the cycle c -> d -> c.
    """
    path = tmp_path / "dcg.dot"
    path.write_text("digraph dct { a -> b -> c -> d; d -> c; }")
    return str(path)


# The pairs of variables that should be independent in the DAG from the dag_path fixture once the edge Z -> Y is added.
# We can't just do "nx.non_edges" here, since some independences are bidirectional (if there is no path from
# X -> ... -> Y) and some are unidirectional (if X -> Y is not in the DAG but X -> ... -> Y is).
EXPECTED_INDEPENDENCES = [
    ("X1", "M"),
    ("X1", "Y"),
    ("X1", "X2"),
    ("X2", "X1"),
    ("X1", "X3"),
    ("X3", "X1"),
    ("Z", "X3"),
    ("X3", "Z"),
    ("X2", "M"),
    ("X2", "Y"),
    ("X3", "Y"),
    ("X2", "X3"),
    ("X3", "X2"),
]


@pytest.mark.parametrize("threads", [0, 2], ids=["sequential", "parallel"])
def test_all_metamorphic_relations_implied_by_dag(dag_path, threads):
    """Test that generate_causal_tests produces a causal test for each edge and an independence test for each
    (minimal) conditional independence implied by the DAG, both sequentially and in parallel.
    """
    dag = CausalDAG(dag_path, datatypes={v: float for v in {"X1", "X2", "X3", "Y", "Z", "M"}})
    dag.add_edge("Z", "Y")  # Add a direct path from Z to Y so M becomes a mediator

    expected_tests = []
    for treatment, outcome in dag.edges:
        expected_tests.append(
            CausalTestCase(
                expected_causal_effect=SomeEffect(),
                effect_measure="coefficient",
                estimator=LinearRegressionEstimator(
                    treatment_variable=treatment,
                    outcome_variable=outcome,
                    adjustment_set=dag.identification(treatment_variable=treatment, outcome_variable=outcome),
                ),
                name=f"{treatment} -> {outcome}",
                skip=False,
            )
        )
    for treatment, outcome in EXPECTED_INDEPENDENCES:
        expected_tests.append(
            CausalTestCase(
                expected_causal_effect=NoEffect(),
                effect_measure="coefficient",
                estimator=LinearRegressionEstimator(
                    treatment_variable=treatment,
                    outcome_variable=outcome,
                    adjustment_set=dag.identification(treatment_variable=treatment, outcome_variable=outcome),
                ),
                name=f"{treatment} _||_ {outcome}",
                skip=False,
            )
        )

    assert sorted(map(lambda t: t.to_dict(), expected_tests), key=sort_test_dict) == sorted(
        map(lambda t: t.to_dict(), dag.generate_causal_tests(threads=threads)), key=sort_test_dict
    )


def test_all_metamorphic_relations_implied_by_dag_no_datatype(dag_path):
    """Test that generate_causal_tests raises an error if a variable has no datatype specified."""
    causal_dag = CausalDAG(dag_path)
    with pytest.raises(ValueError, match=re.escape("No datatype specified for Z.")):
        causal_dag.generate_causal_tests()


def test_all_metamorphic_relations_implied_by_dag_ignore_cycles(dcg_path):
    """Test that generate_causal_tests excludes nodes that are part of a cycle when ignore_cycles is set."""
    dcg = CausalDAG(dcg_path, ignore_cycles=True, datatypes={v: float for v in {"a", "b", "c", "d"}})

    expected_tests = [
        CausalTestCase(
            expected_causal_effect=SomeEffect(),
            effect_measure="coefficient",
            estimator=LinearRegressionEstimator(
                treatment_variable="a",
                outcome_variable="b",
                adjustment_set=dcg.identification(treatment_variable="a", outcome_variable="b"),
            ),
            name="a -> b",
            skip=False,
        )
    ]
    assert sorted(map(lambda t: t.to_dict(), expected_tests), key=sort_test_dict) == sorted(
        map(lambda t: t.to_dict(), dcg.generate_causal_tests(threads=2)), key=sort_test_dict
    )
