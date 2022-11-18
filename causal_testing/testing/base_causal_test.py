from dataclasses import dataclass
from causal_testing.specification.variable import Variable

@dataclass
class BaseCausalTest:
    """
    A base causal test case represents the relationship of an edge on a causal DAG.
    """
    treatment_variable: Variable
    outcome_variable: Variable
    effect: str = 'total'

