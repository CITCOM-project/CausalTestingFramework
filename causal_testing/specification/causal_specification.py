"""This module holds the abstract CausalSpecification data class, which holds a Scenario and CausalDag"""

from dataclasses import dataclass
from typing import Union

from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.specification.scenario import Scenario

Node = Union[str, int]  # Node type hint: A node is a string or an int


@dataclass
class CausalSpecification:
    """
    Data class storing the Causal Specification (combination of Scenario and Causal Dag)
    """

    scenario: Scenario
    causal_dag: CausalDAG

    def __str__(self):
        return f"Scenario: {self.scenario}\nCausal DAG:\n{self.causal_dag}"
