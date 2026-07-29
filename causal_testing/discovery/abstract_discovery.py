"""
This module implements the abstract Discovery class to infer causal DAGs from data.
"""

import random
import re
import warnings
from abc import ABC, abstractmethod
from itertools import permutations

import numpy as np
import pandas as pd
import rustworkx as rx

from causal_testing.causal_testing_framework import CausalTestingFramework
from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.testing.causal_test_result import TestOutcome

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

    def evaluate_tests(self, causal_dag: CausalDAG) -> pd.DataFrame:
        """
        Generate and evaluate causal test cases from the supplied CausalDAG and return a list of edges for which the
        corresponding causal test case failed.
        These results are then assigned to a new attribute `test_results` within the individual for later reuse.

        :param causal_dag: The CausalDAG to evaluate.
        :returns: Pandas dataframe with test outcome details
                  (result, expected effect, treatment, outcome, effect direction).
        """

        ctf = CausalTestingFramework(dag=causal_dag, df=self.df)
        causal_dag.datatypes = self.df.dtypes
        ctf.test_cases = causal_dag.generate_causal_tests()
        causal_dag.test_cases = ctf.test_cases

        results = []

        for test_case in ctf.test_cases:
            try:
                test_case.execute_test(self.df)
                results.append(
                    {
                        "result": (
                            TestOutcome.PASS
                            if test_case.expected_causal_effect.apply(test_case.result.effect_estimate)
                            else TestOutcome.FAIL
                        ),
                        "expected_effect": test_case.expected_causal_effect.__class__.__name__,
                        "treatment": test_case.treatment_variable,
                        "outcome": test_case.outcome_variable,
                    }
                )
            except np.linalg.LinAlgError:
                results.append(
                    {
                        "result": TestOutcome.INESTIMABLE,
                        "expected_effect": test_case.expected_causal_effect.__class__.__name__,
                        "treatment": test_case.treatment_variable,
                        "outcome": test_case.outcome_variable,
                    }
                )

        return pd.DataFrame(results)
