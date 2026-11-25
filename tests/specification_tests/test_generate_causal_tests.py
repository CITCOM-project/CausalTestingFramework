import re
from pathlib import Path

import pytest

from causal_testing.estimation.linear_regression_estimator import LinearRegressionEstimator
from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.testing.causal_effect import NoEffect, SomeEffect
from causal_testing.testing.causal_test_case import CausalTestCase


def sort_test_dict(test: dict):
    return test["name"]


class TestGenerateCausalTestCases:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.dag_dot_path = str(tmp_path / "dag.dot")
        dag_dot = """digraph DAG { rankdir=LR; X1 -> Z; Z -> M; M -> Y; X2 -> Z; X3 -> M;}"""
        Path(self.dag_dot_path).write_text(dag_dot)
        self.dcg_dot_path = str(tmp_path / "dcg.dot")
        dcg_dot = """digraph dct { a -> b -> c -> d; d -> c; }"""
        Path(self.dcg_dot_path).write_text(dcg_dot)
        self.default_control_input_config = {"X1": 1, "X2": 2, "X3": 3}
        self.default_treatment_input_config = {"X1": 2, "X2": 3, "X3": 3}

    def test_all_metamorphic_relations_implied_by_dag(self):
        dag = CausalDAG(self.dag_dot_path, datatypes={v: float for v in {"X1", "X2", "X3", "Y", "Z", "M"}})
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
        for treatment, outcome in [
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
        ]:
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
            map(lambda t: t.to_dict(), dag.generate_causal_tests()), key=sort_test_dict
        )

    def test_all_metamorphic_relations_implied_by_dag_parallel(self):
        dag = CausalDAG(self.dag_dot_path, datatypes={v: float for v in {"X1", "X2", "X3", "Y", "Z", "M"}})
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
        # We can't just do "nx.non_edges" here, since some independences are bidirectional (if there is no path from
        # X -> ... -> Y) and some are unidirectional (if X -> Y is not in the DAG but X -> ... -> Y is).
        for treatment, outcome in [
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
        ]:
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
            map(lambda t: t.to_dict(), dag.generate_causal_tests(threads=2)), key=sort_test_dict
        )

    def test_all_metamorphic_relations_implied_by_dag_no_datatype(self):
        causal_dag = CausalDAG(self.dag_dot_path)
        with pytest.raises(ValueError, match=re.escape("No datatype specified for Z.")):
            causal_dag.generate_causal_tests()

    def test_all_metamorphic_relations_implied_by_dag_ignore_cycles(self):
        dcg = CausalDAG(self.dcg_dot_path, ignore_cycles=True, datatypes={v: float for v in {"a", "b", "c", "d"}})

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
