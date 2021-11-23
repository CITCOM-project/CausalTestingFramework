from abc import ABC
from causal_testing.specification.constraint import NormalDistribution
from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.specification.scenario import Scenario
from typing import Union
import logging

Node = Union[str, int]  # Node type hint: A node is a string or an int
logger = logging.getLogger(__name__)


class CausalSpecification(ABC):

    def __init__(self, scenario: Scenario, causal_dag: CausalDAG):
        self.scenario = scenario
        self.causal_dag = causal_dag

    def __str__(self):
        return f'Scenario: {self.scenario}\nCausal DAG:\n{self.causal_dag}'


if __name__ == "__main__":
    scenario = Scenario({"Vaccine": "Pfizer"})
    age_constraint = NormalDistribution(40, 10)
    scenario.add_constraint("Age", age_constraint)

    causal_dag = CausalDAG()
    causal_dag.add_edge("Vaccine", "Cumulative Infections")
    causal_dag.add_edge("Age", "Vaccine")
    causal_dag.add_edge("Age", "Cumulative Infections")

    causal_specification = CausalSpecification(scenario, causal_dag)
    print(causal_specification)
