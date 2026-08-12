"""
This module implements a set of utilities to help visualise causal test results.
"""

import networkx as nx
import pandas as pd

from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.testing.causal_test_case import CausalTestCase
from causal_testing.testing.causal_test_result import TestOutcome


def results_dag(
    test_cases: list[CausalTestCase],
    dag: CausalDAG,
    output_file: str = None,
    view_independences: bool = True,
    colours: dict[TestOutcome, str] = None,
) -> CausalDAG:
    """
    View causal test results as a graph.

    :param test_cases: The causal tests (with results).
    :param dag: The causal DAG that corresponds to the test results.
    :param output_file: The name of the file to write to.
    :param view_independences: Whether to display failed independence tests.
    :param colours: Optional dictionary of colours to display the test outcomes.
                    By default, pass=green, fail=red, inestimable=orange.
    """
    default_colours = {TestOutcome.PASS: "green", TestOutcome.INESTIMABLE: "orange", TestOutcome.FAIL: "red"}

    if colours is not None:
        colours = default_colours | colours
    else:
        colours = default_colours

    result_dag = nx.DiGraph()
    result_dag.add_nodes_from(dag.nodes)
    result_dag.add_edges_from(dag.edges)

    for test in test_cases:
        if test.result:
            effect_estimate = pd.concat(
                [
                    test.result.effect_estimate.ci_low,
                    test.result.effect_estimate.value,
                    test.result.effect_estimate.ci_high,
                ],
                axis=1,
            )
            effect_estimate.columns = ["ci_low", "estimate", "ci_high"]
            if (test.treatment_variable, test.outcome_variable) in result_dag.edges:
                result_dag[test.treatment_variable][test.outcome_variable]["label"] = test.result.effect_direction()
                result_dag[test.treatment_variable][test.outcome_variable]["color"] = colours[test.result.outcome]
                result_dag[test.treatment_variable][test.outcome_variable]["fontcolor"] = colours[test.result.outcome]

            elif view_independences and test.result.outcome != TestOutcome.PASS:
                result_dag.add_edge(test.treatment_variable, test.outcome_variable, ignore_cycles=True)
                result_dag[test.treatment_variable][test.outcome_variable]["style"] = "dashed"
                result_dag[test.treatment_variable][test.outcome_variable]["label"] = test.result.effect_direction()
                result_dag[test.treatment_variable][test.outcome_variable]["color"] = colours[test.result.outcome]
                result_dag[test.treatment_variable][test.outcome_variable]["fontcolor"] = colours[test.result.outcome]

    if output_file is not None:
        nx.drawing.nx_pydot.write_dot(result_dag, output_file)

    return result_dag
