"""
This module implements the abstract Discovery class to infer causal DAGs from data.
"""

import random
import re
import warnings
from abc import ABC, abstractmethod
from enum import Enum
from itertools import permutations

import networkx as nx
import numpy as np
import pandas as pd
import rustworkx as rx

from causal_testing.main import CausalTestingFramework
from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.specification.scenario import Scenario
from causal_testing.testing.causal_effect import Negative, Positive
from causal_testing.testing.causal_test_result import CausalTestResult
from causal_testing.testing.metamorphic_relation import generate_metamorphic_relations

TestResult = Enum("TestResult", [("PASS", 2), ("FAIL", 0), ("INESTIMABLE", 1)])

# Ignore warnings from statsmodels when we try to evaluate test cases
warnings.simplefilter("ignore")


def simple_cycle(causal_dag: CausalDAG):
    """
    Find a cycle in the given CausalDAG, if one exists, returns the first found.

    :param causal_dag: The CausalDAG to check for cycles.
    :returns: A list of edges in the cycle, or an empty list if there are no cycles.
    """
    rx_graph = rx.networkx_converter(causal_dag)
    return [(rx_graph[i], rx_graph[j]) for i, j in rx.digraph_find_cycle(rx_graph)]


def effect_direction(result: CausalTestResult) -> str:
    """
    Check whether the estimated causal effect is negative or positive.

    :param result: The causal test result object.
    :returns: Whether the estimated causal test is positive or negative (or no effect).
    """
    if pd.api.types.is_numeric_dtype(
        result.estimator.df[result.estimator.base_test_case.treatment_variable.name]
    ) and pd.api.types.is_numeric_dtype(result.estimator.df[result.estimator.base_test_case.outcome_variable.name]):
        if Negative().apply(result):
            return "negative"
        if Positive().apply(result):
            return "positive"
    return None


def is_match(u: str, v: str, patterns: list[str]):
    """
    Check whether a given edge matches a given pattern.

    :param u: The origin node of the edge.
    :param v: The destination node of the edge.
    :param patterns: A list of tuples containing the patterns to check against.
    :returns: True if the edge matches the pattern, False otherwise.
    """
    return any(re.fullmatch(pat_u, u) and re.fullmatch(pat_v, v) for pat_u, pat_v in patterns)


class Discovery(ABC):
    """
    Abstract class for causal discovery.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        random_seed: int = 0,
        exclude_edges: str = None,
        include_edges: str = None,
        alpha: float = 0.05,
    ):

        random.seed(random_seed)
        self.df = df
        self.random_seed = int(random_seed)
        self.alpha = float(alpha)

        self.possible_edges = []
        self.include_edges = []
        self.exclude_edges = []

        for u, v in permutations(df.columns, 2):
            if exclude_edges and is_match(u, v, exclude_edges):
                self.exclude_edges.append((u, v))
            else:
                self.possible_edges.append((u, v))

            if include_edges and is_match(u, v, include_edges):
                self.include_edges.append((u, v))

        if self.include_edges:
            # Check to make sure that the include edges don't specify a cycle
            initial_dag = CausalDAG()
            initial_dag.add_edges_from(self.include_edges)

            if not initial_dag.is_acyclic():
                raise ValueError(
                    "Specified include edges include a cycle, making it impossible to infer a DAG. "
                    "Please resolve this and try again."
                )

    @abstractmethod
    def discover(self) -> CausalDAG:
        """
        Discover the causal DAG.

        :returns: The inferred causal DAG.
        """

    def remove_cycles(self, causal_dag: CausalDAG):
        """
        Remove cycles from individuals by iteratively deleting a random edge from each cycle until there are no more
        cycles.

        :param causal_dag: The CausalDAG to be repaired.
        """
        nodes = causal_dag.nodes
        cycle = simple_cycle(causal_dag)
        while cycle:
            idx = random.choice(range(len(cycle)))
            while cycle[idx] in self.include_edges:
                idx = (idx + 1) % len(cycle)
            causal_dag.remove_edge(cycle[idx][0], cycle[idx][1])
            cycle = simple_cycle(causal_dag)
        causal_dag.add_nodes_from(nodes)

    def write_dot(self, individual: CausalDAG, output_file: str):
        """
        Write the given individual to the given output file.

        :param individual: The causal DAG to output.
        :param output_file: The name of the file to write to.
        """
        if hasattr(individual, "test_results"):
            for _, test in individual.test_results.iterrows():
                if (test["treatment"], test["outcome"]) in individual.edges:
                    if test["result"] == TestResult.PASS:
                        individual[test["treatment"]][test["outcome"]]["color"] = "green"
                    elif test["result"] == TestResult.INESTIMABLE:
                        individual[test["treatment"]][test["outcome"]]["color"] = "orange"
                    elif test["result"] == TestResult.FAIL:
                        individual[test["treatment"]][test["outcome"]]["color"] = "red"
                    else:
                        raise ValueError(f"Invalid test outcome {test['result']}")
                else:
                    individual.add_edge(test["treatment"], test["outcome"], ignore_cycles=True)
                    individual[test["treatment"]][test["outcome"]]["style"] = "dashed"
                    if test["result"] == TestResult.PASS:
                        individual[test["treatment"]][test["outcome"]]["style"] = "invis"
                        individual[test["treatment"]][test["outcome"]]["constraint"] = False
                    elif test["result"] == TestResult.INESTIMABLE:
                        individual[test["treatment"]][test["outcome"]]["color"] = "orange"
                    elif test["result"] == TestResult.FAIL:
                        individual[test["treatment"]][test["outcome"]]["color"] = "red"
                    else:
                        raise ValueError(f"Invalid test outcome {test['result']}")

        nx.drawing.nx_pydot.write_dot(individual, output_file)

    def _json_stub_params(self, outcome: str) -> str:
        if pd.api.types.is_bool_dtype(self.df[outcome]):
            return {"estimator": "LogisticRegressionEstimator", "estimate_type": "unit_odds_ratio"}
        if pd.api.types.is_categorical_dtype(self.df[outcome]) or pd.api.types.is_object_dtype(self.df[outcome]):
            return {"estimator": "MultinomialRegressionEstimator", "estimate_type": "unit_odds_ratio"}
        if pd.api.types.is_numeric_dtype(self.df[outcome]):
            return {"estimator": "LinearRegressionEstimator", "estimate_type": "coefficient"}
        raise ValueError(f"Invalid datatype {self.df.dtypes[outcome]}")

    def evaluate_tests(self, causal_dag: CausalDAG) -> pd.DataFrame:
        """
        Generate and evaluate causal test cases from the supplied CausalDAG and return a list of edges for which the
        corresponding causal test case failed.
        These results are then assigned to a new attribute `test_results` within the individual for later reuse.

        :param causal_dag: The CausalDAG to evaluate.
        :returns: Pandas dataframe with test outcome details
                  (result, expected effect, treatment, outcome, effect direction).
        """

        ctf = CausalTestingFramework(None)
        ctf.dag = causal_dag
        ctf.data = self.df
        ctf.create_variables()
        ctf.scenario = Scenario(list(ctf.variables["inputs"].values()) + list(ctf.variables["outputs"].values()))

        ctf.test_cases = ctf.create_test_cases(
            {
                "tests": [
                    relation.to_json_stub(
                        alpha=self.alpha,
                        **self._json_stub_params(relation.base_test_case.outcome_variable),
                    )
                    for relation in generate_metamorphic_relations(causal_dag)
                ]
            }
        )

        results = []

        for test_case, result in zip(ctf.test_cases, ctf.test_cases):
            try:
                result = test_case.execute_test()
                results.append(
                    {
                        "result": (
                            TestResult.PASS if test_case.expected_causal_effect.apply(result) else TestResult.FAIL
                        ),
                        "expected_effect": test_case.expected_causal_effect.__class__.__name__,
                        "treatment": test_case.base_test_case.treatment_variable.name,
                        "outcome": test_case.base_test_case.outcome_variable.name,
                        "effect": effect_direction(result),
                    }
                )
            except np.linalg.LinAlgError:
                results.append(
                    {
                        "result": TestResult.INESTIMABLE,
                        "expected_effect": test_case.expected_causal_effect.__class__.__name__,
                        "treatment": test_case.base_test_case.treatment_variable.name,
                        "outcome": test_case.base_test_case.outcome_variable.name,
                    }
                )

        causal_dag.test_results = pd.DataFrame(results)
        return pd.DataFrame(results)
