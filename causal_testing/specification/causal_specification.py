import logging
from abc import ABC
from typing import Union

from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.specification.scenario import Scenario

Node = Union[str, int]  # Node type hint: A node is a string or an int
logger = logging.getLogger(__name__)


class CausalSpecification(ABC):
    """
    Abstract Class for the Causal Specification (combination of Scenario and Causal Dag)
    """
    def __init__(self, scenario: Scenario, causal_dag: CausalDAG):
        self.scenario = scenario
        self.causal_dag = causal_dag

    def __str__(self):
        return f"Scenario: {self.scenario}\nCausal DAG:\n{self.causal_dag}"
