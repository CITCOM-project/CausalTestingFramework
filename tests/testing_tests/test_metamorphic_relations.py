import unittest
import os
import shutil, tempfile
import json
import networkx as nx

from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.specification.scenario import Scenario
from causal_testing.specification.variable import Input, Output
from causal_testing.testing.base_test_case import BaseTestCase
from causal_testing.testing.causal_test_case import CausalTestCase
from causal_testing.testing.causal_effect import NoEffect, SomeEffect
from causal_testing.estimation.abstract_estimator import Estimator
from causal_testing.estimation.linear_regression_estimator import LinearRegressionEstimator
from causal_testing.estimation.logistic_regression_estimator import LogisticRegressionEstimator
from causal_testing.estimation.multinomial_regression_estimator import MultinomialRegressionEstimator


class TestMetamorphicRelation(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir_path = tempfile.mkdtemp()
        self.dag_dot_path = os.path.join(self.temp_dir_path, "dag.dot")
        dag_dot = """digraph DAG { rankdir=LR; X1 -> Z; Z -> M; M -> Y; X2 -> Z; X3 -> M;}"""
        with open(self.dag_dot_path, "w") as f:
            f.write(dag_dot)
        self.dcg_dot_path = os.path.join(self.temp_dir_path, "dcg.dot")
        dcg_dot = """digraph dct { a -> b -> c -> d; d -> c; }"""
        with open(self.dcg_dot_path, "w") as f:
            f.write(dcg_dot)

        X1 = Input("X1", float)
        X2 = Input("X2", float)
        X3 = Input("X3", float)
        Z = Output("Z", float)
        M = Output("M", float)
        Y = Output("Y", float)
        self.scenario = Scenario(variables={X1, X2, X3, Z, M, Y})
        self.default_control_input_config = {"X1": 1, "X2": 2, "X3": 3}
        self.default_treatment_input_config = {"X1": 2, "X2": 3, "X3": 3}

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir_path)

    def test_all_metamorphic_relations_implied_by_dag(self):
        dag = CausalDAG(self.dag_dot_path, datatypes={v: float for v in {"X1", "X2", "X3", "Y", "Z", "M"}})
        dag.add_edge("Z", "Y")  # Add a direct path from Z to Y so M becomes a mediator

        expected_tests = []
        for treatment, outcome in dag.edges:
            base_test_case = BaseTestCase(Input(treatment, float), Output(outcome, float))
            expected_tests.append(
                CausalTestCase(
                    base_test_case=base_test_case,
                    expected_causal_effect=SomeEffect(),
                    estimate_type="coefficient",
                    estimator=LinearRegressionEstimator(base_test_case),
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
            base_test_case = BaseTestCase(Input(treatment, float), Output(outcome, float))
            expected_tests.append(
                CausalTestCase(
                    base_test_case=base_test_case,
                    expected_causal_effect=NoEffect(),
                    estimate_type="coefficient",
                    estimator=LinearRegressionEstimator(base_test_case),
                    name=f"{treatment} -> {outcome}",
                    skip=False,
                )
            )

        self.assertEqual(sorted(map(str, expected_tests)), sorted(map(str, dag.generate_causal_tests())))

    def test_all_metamorphic_relations_implied_by_dag_parallel(self):
        dag = CausalDAG(self.dag_dot_path, datatypes={v: float for v in {"X1", "X2", "X3", "Y", "Z", "M"}})
        dag.add_edge("Z", "Y")  # Add a direct path from Z to Y so M becomes a mediator

        expected_tests = []
        for treatment, outcome in dag.edges:
            base_test_case = BaseTestCase(Input(treatment, float), Output(outcome, float))
            expected_tests.append(
                CausalTestCase(
                    base_test_case=base_test_case,
                    expected_causal_effect=SomeEffect(),
                    estimate_type="coefficient",
                    estimator=LinearRegressionEstimator(base_test_case),
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
            base_test_case = BaseTestCase(Input(treatment, float), Output(outcome, float))
            expected_tests.append(
                CausalTestCase(
                    base_test_case=base_test_case,
                    expected_causal_effect=NoEffect(),
                    estimate_type="coefficient",
                    estimator=LinearRegressionEstimator(base_test_case),
                    name=f"{treatment} -> {outcome}",
                    skip=False,
                )
            )

        self.assertEqual(sorted(map(str, expected_tests)), sorted(map(str, dag.generate_causal_tests(threads=2))))

    def test_all_metamorphic_relations_implied_by_dag_ignore_cycles(self):
        dcg = CausalDAG(self.dcg_dot_path, ignore_cycles=True, datatypes={v: float for v in {"a", "b", "c", "d"}})

        base_test_case = BaseTestCase(Input("a", float), Output("b", float))
        expected_tests = [
            CausalTestCase(
                base_test_case=base_test_case,
                expected_causal_effect=SomeEffect(),
                estimate_type="coefficient",
                estimator=LinearRegressionEstimator(base_test_case),
                name=f"a -> b",
                skip=False,
            )
        ]
        self.assertEqual(sorted(map(str, expected_tests)), sorted(map(str, dcg.generate_causal_tests(threads=2))))
