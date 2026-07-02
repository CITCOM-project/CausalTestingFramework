"""
This module implements a hill climbing algorithm to optimise causal DAGs based on the tests that pass/fail.
"""

import random
import time
import warnings
from itertools import permutations
from enum import Enum

import networkx as nx
import rustworkx as rx
import numpy as np
import pandas as pd

from causal_testing.main import CausalTestingFramework
from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.specification.scenario import Scenario
from causal_testing.testing.causal_test_result import CausalTestResult
from causal_testing.testing.causal_effect import Positive, Negative
from causal_testing.testing.metamorphic_relation import generate_metamorphic_relations

TestResult = Enum("TestResult", [("PASS", "pass"), ("FAIL", "fail"), ("INESTIMABLE", "inestimable")])

warnings.simplefilter("ignore")

# lexicographical order (max pass, minimise failure, minimise unknown)
#  e.g. (X pass, Y fail, Z+1 unknown) is better than (X pass, Y+1 fail, Z unknown)


def simple_cycle(causal_dag: CausalDAG):
    """
    Find a cycle in the given CausalDAG, if one exists, returns the first found.

    :param causal_dag: The CausalDAG to check for cycles.
    :returns: A list of edges in the cycle, or an empty list if there are no cycles.
    """
    rx_graph = rx.networkx_converter(causal_dag)
    return [(rx_graph[i], rx_graph[j]) for i, j in rx.digraph_find_cycle(rx_graph)]


def remove_cycles(causal_dag: CausalDAG, included_edges: set):
    """
    Remove cycles from individuals by iteratively deleting a random edge from each cycle until there are no more cycles.

    :param causal_dag: The CausalDAG to be repaired.
    :param included_edges: A set of edges that must be included in the repaired DAG.
    """
    nodes = causal_dag.nodes
    cycle = simple_cycle(causal_dag)
    while cycle:
        idx = random.choice(range(len(cycle)))
        while cycle[idx] in included_edges:
            idx = (idx + 1) % len(cycle)
        causal_dag.remove_edge(cycle[idx][0], cycle[idx][1])
        cycle = simple_cycle(causal_dag)
    causal_dag.add_nodes_from(nodes)


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


def evaluate_tests(causal_dag: CausalDAG, df: pd.DataFrame):
    """
    Generate and evaluate causal test cases from the supplied CausalDAG and return a list of edges for which the
    corresponding causal test case failed.
    These results are then assigned to a new attribute `test_results` within the individual for later reuse.

    :param causal_dag: The CausalDAG to evaluate.
    :param df: The data with which to evaluate the causal test cases.
    """

    ctf = CausalTestingFramework(None)
    ctf.dag = causal_dag
    ctf.data = df
    ctf.create_variables()
    ctf.scenario = Scenario(list(ctf.variables["inputs"].values()) + list(ctf.variables["outputs"].values()))
    ctf.test_cases = ctf.create_test_cases(
        {
            "tests": [
                relation.to_json_stub(
                    estimator="LogisticRegressionEstimator", estimate_type="unit_odds_ratio", alpha=0.01
                )
                for relation in generate_metamorphic_relations(causal_dag)
            ]
        }
    )
    results = []

    for test_case, result in zip(ctf.test_cases, ctf.run_tests(silent=True)):
        if result.effect_estimate is None:
            results.append(
                {
                    "result": TestResult.INESTIMABLE,
                    "expected_effect": test_case.expected_causal_effect.__class__.__name__,
                    "treatment": test_case.base_test_case.treatment_variable.name,
                    "outcome": test_case.base_test_case.outcome_variable.name,
                }
            )
        else:
            results.append(
                {
                    "result": TestResult.PASS if test_case.expected_causal_effect.apply(result) else TestResult.FAIL,
                    "expected_effect": test_case.expected_causal_effect.__class__.__name__,
                    "treatment": test_case.base_test_case.treatment_variable.name,
                    "outcome": test_case.base_test_case.outcome_variable.name,
                    "effect": effect_direction(result),
                }
            )

    causal_dag.test_results = pd.DataFrame(results)


# MF TODO: Double check whether this method is actually necessary.
def normalised_counts(test_results: pd.DataFrame) -> dict:
    """
    Normalise the absolute numbers of pass/fail/inestimable test outcomes.
    MF Note 2026-06-15: I can't actually remember what this method was supposed to do. I need to double check it.
    :param test_results: Dataframe containing the raw pass/fail/inestimable outcome of each test case.
    :returns: Dictionary containing the number of pass/fail/inestimable outcomes, normalised by dividing by the total number
              of each.
    """
    counts = pd.concat(
        [
            pd.DataFrame(np.sort(test_results[["treatment", "outcome"]], axis=1), columns=["treatment", "outcome"]),
            pd.get_dummies(test_results["result"]).astype(int),
        ],
        axis=1,
    )
    for col in TestResult:
        if col not in counts.columns:
            counts[col] = 0
    counts = counts.groupby(["treatment", "outcome"]).sum().reset_index()[list(TestResult)]
    counts = counts.apply(lambda col: col / counts.sum(axis=1))
    return counts.sum(axis=0).to_dict()


def evaluate_fitness_tier(
    individual: CausalDAG,
    df: pd.DataFrame,
) -> tuple[tuple[float, float, float], list[tuple[str, str]]]:
    """
    Evaluate the fitness of a given causal DAG by evaluating the corresponding test cases using a tier based
    fitness metric.

    :param individual: The candidate individual to evaluate.
    :param df: The data with which to evaluate the causal tests.
    :returns: Tuple of the form (X, Y), where X is a triple containing the number of passing, failing, and inestimable
              tests respectively, and Y is a list of failing edges.
    """
    evaluate_tests(individual, df)
    counts = normalised_counts(individual.test_results)

    problem_tests = individual.test_results.loc[individual.test_results["result"] != TestResult.PASS]
    problem_edges = problem_tests[["treatment", "outcome"]].apply(tuple, axis=1).tolist()
    problem_edges.extend(
        problem_tests.query("expected_effect == 'NoEffect'")[["outcome", "treatment"]].apply(tuple, axis=1).tolist()
    )

    fitness_values = (
        counts.get(TestResult.PASS, 0),
        -counts.get(TestResult.FAIL, 0),
        -counts.get(TestResult.INESTIMABLE, 0),
    )
    print(f"({fitness_values[0]}, {-fitness_values[1]}, {-fitness_values[2]})")
    return fitness_values, problem_edges


def evaluate_fitness_score(
    individual: CausalDAG, df: pd.DataFrame
) -> tuple[tuple[float, float, float], list[tuple[str, str]]]:
    """
    Evaluate the fitness of a given causal DAG by evaluating the corresponding test cases using a score based
    fitness metric.

    :param individual: The candidate individual to evaluate.
    :param df: The data with which to evaluate the causal tests.
    :returns: Tuple of the form (X, Y), where X is the fitness score, and Y is a list of failing edges.
    """
    evaluate_tests(individual, df)
    counts = normalised_counts(individual.test_results)

    problem_tests = individual.test_results.query("result != TestResult.PASS")
    problem_edges = problem_tests[["treatment", "outcome"]].apply(tuple, axis=1).tolist()
    problem_edges.extend(
        problem_tests.query("expected_effect == 'NoEffect'")[["outcome", "treatment"]].apply(tuple, axis=1).tolist()
    )

    new_fitness_values = counts.get(TestResult.PASS, 0) * 2 + counts.get(TestResult.INESTIMABLE, 0) * 1
    print(" ", f"{new_fitness_values} / {sum(counts.values()) * 2}")
    return new_fitness_values, problem_edges


def write_dot(individual: CausalDAG, output_file: str):
    """
    Write the given individual to the given output file.
    :param individual: The causal DAG to output.
    :param output_file: The name of the file to write to.
    """
    if hasattr(individual, "test_results"):
        print(individual.test_results)
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


def evolve_dag(
    df: pd.DataFrame,
    random_seed: int = 0,
    output_file: str = None,
    include_edges_file: str = None,
    exclude_edges_file: str = None,
    fitness_function: callable = evaluate_fitness_tier,
    max_iterations: int = None,
    max_iterations_without_improvement: int = None,
) -> CausalDAG:
    """
    Evolve a causal DAG for a given dataset.
    :param df: The data for which to fit a causal DAG.
    :param random_seed: The random seed to use for genetic computation.
    :param output_file: Where to save the inferred causal DAG (if supplied).
    :param include_edges_file: Path to file containing edges to include.
    :param exclude_edges_file: Path to file containing edges to exclude.
    :returns: The inferred causal DAG.
    """
    random.seed(random_seed)

    included_edges = set(nx.nx_pydot.read_dot(include_edges_file).edges()) if include_edges_file is not None else set()
    excluded_edges = set(nx.nx_pydot.read_dot(exclude_edges_file).edges()) if exclude_edges_file is not None else set()
    possible_edges = sorted(list((u, v) for u, v in permutations(df.columns, 2) if (u, v) not in excluded_edges))

    start_time = time.time()
    individual = CausalDAG()
    individual.add_nodes_from(df.columns)
    individual.add_edges_from(possible_edges)
    remove_cycles(individual, included_edges)
    fitness_values, problem_edges = fitness_function(individual, df)

    if max_iterations is None:
        max_iterations = 100
    if max_iterations_without_improvement is None:
        max_iterations_without_improvement = 20

    iterations = max_iterations
    iterations_without_improvement = 0

    while problem_edges and iterations and iterations_without_improvement < max_iterations_without_improvement:
        iterations -= 1
        if fitness_function == evaluate_fitness_tier:
            print(
                iterations,
                f"({fitness_values[0]}, {-fitness_values[1]}, {-fitness_values[2]})",
                iterations_without_improvement,
            )
        else:
            print(iterations, f"({fitness_values})", iterations_without_improvement)

        new_individual = individual.copy()
        for origin, dest in random.sample(
            problem_edges + (possible_edges if iterations_without_improvement > 10 else []),
            random.randint(1, len(problem_edges)),
        ):
            if new_individual.has_edge(origin, dest) and (origin, dest) not in included_edges:
                new_individual.remove_edge(origin, dest)
            elif not new_individual.has_edge(origin, dest) and (origin, dest) not in excluded_edges:
                # Want to bypass the cycle check of CausalDAG as we remove the cycles afterwards
                new_individual.add_edge(origin, dest, ignore_cycles=True)
        remove_cycles(new_individual, included_edges)
        new_fitness_values, new_problem_edges = fitness_function(new_individual, df)

        if new_fitness_values > fitness_values:
            fitness_values = new_fitness_values
            problem_edges = new_problem_edges
            individual = new_individual
            iterations_without_improvement = 0
        else:
            iterations_without_improvement += 1

    end_time = time.time()
    individual.graph["fitness"] = fitness_values
    individual.graph["time"] = round(end_time - start_time)

    if output_file is not None:
        write_dot(individual, output_file)
    return individual
