"""
This module contains code to measure various aspects of causal test adequacy.
"""
from causal_testing.testing.causal_test_suite import CausalTestSuite
from causal_testing.specification.causal_specification import CausalSpecification
from causal_testing.testing.estimators import Estimator
from causal_testing.testing.causal_test_case import CausalTestCase
from itertools import combinations


class CausalTestAdequacy:
    def __init__(
        self,
        causal_specification: CausalSpecification,
        test_suite: CausalTestSuite,
    ):
        self.causal_dag, self.scenario = (
            causal_specification.causal_dag,
            causal_specification.scenario,
        )
        self.test_suite = test_suite
        self.estimator = estimator
        self.dag_adequacy = DAGAdequacy(causal_specification, test_suite)


class DAGAdequacy:
    def __init__(
        self,
        causal_specification: CausalSpecification,
        test_suite: CausalTestSuite,
    ):
        self.causal_dag = causal_specification.causal_dag
        self.test_suite = test_suite
        self.tested_pairs = {
            (t.base_test_case.treatment_variable, t.base_test_case.outcome_variable) for t in self.causal_test_suite
        }
        self.pairs_to_test = set(combinations(dag.graph.nodes, 2))
        self.untested_edges = pairs_to_test.difference(tested_pairs)
        self.dag_adequacy = len(tested_pairs) / len(pairs_to_test)
