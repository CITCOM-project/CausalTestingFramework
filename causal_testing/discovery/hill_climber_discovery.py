"""
This module implements a hill climbing algorithm to optimise causal DAGs based on the tests that pass/fail.
"""

import random

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from causal_testing.discovery.abstract_discovery import Discovery
from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.testing.causal_test_result import TestOutcome


class HillClimberDiscovery(Discovery):
    """
    Simple hill climber evolution of cauasl DAGs via 1+1EA.
    Attempts to maximise the number of passing tests and minimise the number of failing tests.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        random_seed: int = 0,
        include_edges: str = None,
        exclude_edges: str = None,
        alpha: float = 0.05,
        max_iterations: int = 30,
        max_iterations_without_improvement: int = 10,
    ):
        super().__init__(
            df=df,
            random_seed=random_seed,
            include_edges=include_edges,
            exclude_edges=exclude_edges,
            alpha=alpha,
        )
        self.max_iterations = int(max_iterations)
        self.max_iterations_without_improvement = int(max_iterations_without_improvement)

    def evaluate_fitness(
        self,
        individual: CausalDAG,
    ) -> tuple[tuple[float, float, float], list[tuple[str, str]]]:
        """
        Evaluate the fitness of a given causal DAG by evaluating the corresponding test cases using a tier based
        fitness metric.
        lexicographical order (max pass, minimise failure, minimise unknown)
        e.g. (X pass, Y fail, Z+1 unknown) is better than (X pass, Y+1 fail, Z unknown)

        :param individual: The candidate individual to evaluate.
        :returns: Tuple of the form (X, Y), where X is a triple containing the number of passing, failing, and
                  inestimable tests respectively, and Y is a list of failing edges.
        """
        self.evaluate_tests(individual)

        # Add extra "var1" and "var2" columns to serve as order independent "treatment" and "outcome"
        query_df = pd.concat(
            [
                individual.test_results,
                pd.DataFrame(
                    np.sort(individual.test_results[["treatment", "outcome"]], axis=1), columns=["var1", "var2"]
                ),
            ],
            axis=1,
        )
        problem_tests = query_df.groupby(["var1", "var2"]).filter(
            # Groups are problematic if at least one test fails or no test passes
            lambda group: (group["result"] == TestOutcome.FAIL).any()
            or ~(group["result"] == TestOutcome.PASS).any()
        )
        problem_edges = problem_tests[["treatment", "outcome"]].apply(tuple, axis=1).tolist()

        counts = {key: len(group) for key, group in query_df.groupby(["result", "expected_effect"], sort=False)}

        no_effect_normalisation = len(list(nx.non_edges(individual))) or 1
        some_effect_normalisation = len(individual.edges) or 1

        fitness_values = (
            (counts.get((TestOutcome.PASS, "NoEffect"), 0)) / no_effect_normalisation,
            -(counts.get((TestOutcome.FAIL, "NoEffect"), 0)) / no_effect_normalisation,
            (counts.get((TestOutcome.PASS, "SomeEffect"), 0)) / some_effect_normalisation,
            -(counts.get((TestOutcome.FAIL, "SomeEffect"), 0)) / some_effect_normalisation,
            -(counts.get((TestOutcome.INESTIMABLE, "NoEffect"), 0)) / no_effect_normalisation,
            -(counts.get((TestOutcome.INESTIMABLE, "SomeEffect"), 0)) / some_effect_normalisation,
        )
        return fitness_values, problem_edges

    def discover(self, individual: CausalDAG = None) -> CausalDAG:
        """
        Discover the causal DAG.

        :param individual: An initial individual for the hill climber to start from
                           (defaults to a fully connected graph).

        :returns: The inferred causal DAG.
        """

        if individual is None:
            individual = CausalDAG(ignore_cycles=True)
            individual.add_nodes_from(self.df.columns)
            for treatment, outcome in self.possible_edges:
                if (treatment, outcome) in self.include_edges:
                    individual.add_edge(treatment, outcome)
        self.remove_cycles(individual)
        fitness_values, problem_edges = self.evaluate_fitness(individual)

        iterations_without_improvement = 0

        for _ in tqdm(range(self.max_iterations)):
            if not problem_edges:
                break

            new_individual = individual.copy()
            for origin, dest in random.sample(
                # If we've gone over the maximum iterations without improvement
                problem_edges
                + (
                    self.possible_edges
                    if iterations_without_improvement > self.max_iterations_without_improvement
                    else []
                ),
                random.randint(1, len(problem_edges)),
            ):
                if new_individual.has_edge(origin, dest) and (origin, dest) not in self.include_edges:
                    new_individual.remove_edge(origin, dest)
                elif not new_individual.has_edge(origin, dest) and (origin, dest) not in self.exclude_edges:
                    # Want to bypass the cycle check of CausalDAG as we remove the cycles afterwards
                    new_individual.add_edge(origin, dest)
            self.remove_cycles(new_individual)
            new_fitness_values, new_problem_edges = self.evaluate_fitness(new_individual)

            if new_fitness_values > fitness_values:
                fitness_values = new_fitness_values
                problem_edges = new_problem_edges
                individual = new_individual
                iterations_without_improvement = 0
            else:
                iterations_without_improvement += 1

        return individual
